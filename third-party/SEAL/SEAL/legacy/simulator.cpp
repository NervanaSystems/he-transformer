#include <algorithm>
#include "simulator.h"
#include "utilities.h"
#include "util/uintarith.h"
#include "util/polyarith.h"
#include <cmath>

using namespace std;
using namespace seal::util;

namespace seal
{
    namespace
    {
        bool are_poly_coefficients_less_than(const BigPoly &poly, const BigUInt &max_coeff)
        {
            return util::are_poly_coefficients_less_than(poly.pointer(), poly.coeff_count(), poly.coeff_uint64_count(), max_coeff.pointer(), max_coeff.uint64_count());
        }

        uint64_t exponentiate_uint(uint64_t base, uint64_t exponent)
        {
            if (exponent == 0)
            {
                return 1;
            }
            uint64_t result = base;
            uint64_t prev_result = 0;
            for (int i = 0; i < exponent - 1; i++)
            {
                if (result < prev_result)
                {
                    throw invalid_argument("uint64 overflow");
                }
                prev_result = result;
                result *= base;
            }
            return result;
        }
    }

    bool Simulation::decrypts(int budget_gap) const
    {
        if (budget_gap < 0)
        {
            throw invalid_argument("budget_gap cannot be negative");
        }
        return (invariant_noise_budget() > budget_gap);
    }

    Simulation::Simulation(const EncryptionParameters &parms, int ciphertext_size, int noise_budget) :
        parms_(parms),
        ciphertext_size_(ciphertext_size)
    {
        int coeff_sig_bit_count = parms_.coeff_modulus().bit_count();

        // Verify parameters
        if (noise_budget < 0 || noise_budget >= coeff_sig_bit_count - 1)
        {
            throw invalid_argument("noise_budget is not in the valid range");
        }
        if (ciphertext_size < 2)
        {
            throw invalid_argument("ciphertext_size must be greater than or equal to 2");
        }

        // Set the noise (scaled by coeff_modulus) to have given noise budget
        // noise_ = 2^(coeff_sig_bit_count - noise_budget - 1) - 1
        int noise_sig_bit_count = coeff_sig_bit_count - noise_budget - 1;
        noise_.resize(coeff_sig_bit_count);
        noise_[0] = 1;
        left_shift_uint(noise_.pointer(), noise_sig_bit_count, noise_.uint64_count(), noise_.pointer());
        decrement_uint(noise_.pointer(), noise_.uint64_count(), noise_.pointer());
    }

    Simulation::Simulation(const EncryptionParameters &parms, int ciphertext_size, const BigUInt &noise) :
        parms_(parms), noise_(noise), 
        ciphertext_size_(ciphertext_size)
    {
    }

    Simulation SimulationEvaluator::add(const Simulation &simulation1, const Simulation &simulation2)
    {
        if (simulation1.parms_ != simulation2.parms_)
        {
            throw invalid_argument("mismatch in encryption parameters");
        }

        int max_ciphertext_size = max(simulation1.ciphertext_size_, simulation2.ciphertext_size_);

        BigUInt result_noise = simulation1.noise_ + simulation2.noise_;

        // If noise overflowed, set it to max
        BigUInt max_noise = simulation1.parms_.coeff_modulus() >> 1;
        if (result_noise > max_noise)
        {
            result_noise = max_noise;
        }
        result_noise.resize(simulation1.parms_.coeff_modulus().bit_count());

        return Simulation(simulation1.parms_, max_ciphertext_size, result_noise);
    }

    Simulation SimulationEvaluator::add_many(const vector<Simulation> &simulations)
    {
        if (simulations.empty())
        {
            throw invalid_argument("simulations vector cannot be empty");
        }
        for (size_t i = 0; i < simulations.size(); ++i)
        {
            if (simulations[i].parms_ != simulations[0].parms_)
            {
                throw invalid_argument("mismatch in encryption parameters");
            }
        }

        // Find the largest of the noises
        BigUInt result_noise = simulations[0].noise_;
        int largest_ciphertext_size = simulations[0].ciphertext_size_;
        for (size_t i = 1; i < simulations.size(); i++)
        {
            if (simulations[i].ciphertext_size_ > largest_ciphertext_size)
            {
                largest_ciphertext_size = simulations[i].ciphertext_size_;
            }
            result_noise += simulations[i].noise_;
        }

        // If noise overflowed, set it to max
        BigUInt max_noise = simulations[0].parms_.coeff_modulus() >> 1;
        if (result_noise > max_noise)
        {
            result_noise = max_noise;
        }
        result_noise.resize(simulations[0].parms_.coeff_modulus().bit_count());

        return Simulation(simulations[0].parms_, largest_ciphertext_size, result_noise);
    }

    Simulation SimulationEvaluator::sub(const Simulation &simulation1, const Simulation &simulation2)
    {
        return add(simulation1, simulation2);
    }

    Simulation SimulationEvaluator::relinearize(const Simulation &simulation, int destination_size)
    {
        //check destination_size is reasonable
        if (destination_size < 2 || destination_size > simulation.ciphertext_size_)
        {
            throw invalid_argument("cannot relinearize to destination_size");
        }

        // Determine number of relinearize_one_step calls which would be needed
        uint64_t relinearize_one_step_calls = static_cast<uint64_t>(simulation.ciphertext_size_ - destination_size);
        if (relinearize_one_step_calls == 0)
        {
            return simulation;
        }

        int poly_modulus_degree = simulation.parms_.poly_modulus().coeff_count() - 1;

        // Noise is ~ old + 2 * min(B, 6*sigma) * t * n * (ell+1) * w * relinearize_one_step_calls

        // First t
        BigUInt result_noise = BigUInt(simulation.parms_.plain_modulus().bit_count(), *simulation.parms_.plain_modulus().pointer());
        
        // Multiply by w
        result_noise <<= simulation.parms_.decomposition_bit_count();

        // Multiply by rest
        int ell = divide_round_up(simulation.parms_.coeff_modulus().bit_count(), simulation.parms_.decomposition_bit_count());
        result_noise *= 2 * static_cast<uint64_t>(min(simulation.parms_.noise_max_deviation(), simulation.parms_.noise_standard_deviation() * 6))
            * poly_modulus_degree * (ell + 1) * relinearize_one_step_calls;

        // Add to existing noise
        result_noise += simulation.noise_;

        // If noise overflowed, set it to max
        BigUInt max_noise = simulation.parms_.coeff_modulus() >> 1;
        if (result_noise > max_noise)
        {
            result_noise = max_noise;
        }
        result_noise.resize(simulation.parms_.coeff_modulus().bit_count());

        return Simulation(simulation.parms_, destination_size, result_noise);
    }

    Simulation SimulationEvaluator::multiply(const Simulation &simulation1, const Simulation &simulation2)
    {
        // Verify that both simulations have the same encryption parameters
        if (simulation1.parms_ != simulation2.parms_)
        {
            throw invalid_argument("mismatch in encryption parameters");
        }

        int poly_modulus_degree = simulation1.parms_.poly_modulus().coeff_count() - 1;

        // Determine new size
        int result_ciphertext_size = simulation1.ciphertext_size_ + simulation2.ciphertext_size_ - 1;

        // Noise is ~ t * sqrt(3n) * [ (12n)^(j1/2)*noise2 + (12n)^(j2/2)*noise1 + (12n)^((j1+j2)/2) ]

        // First compute sqrt(12n) (rounding up) and the powers needed
        uint64_t sqrt_factor_base = static_cast<uint64_t>(ceil(sqrt(static_cast<double>(12 * poly_modulus_degree))));
        uint64_t sqrt_factor_1 = exponentiate_uint(sqrt_factor_base, simulation1.ciphertext_size_ - 1);
        uint64_t sqrt_factor_2 = exponentiate_uint(sqrt_factor_base, simulation2.ciphertext_size_ - 1);
        uint64_t sqrt_factor_total = exponentiate_uint(sqrt_factor_base, simulation1.ciphertext_size_ - 1 + simulation2.ciphertext_size_ - 1);
    
        // Compute also sqrt(3n)
        uint64_t leading_sqrt_factor = static_cast<uint64_t>(ceil(sqrt(static_cast<double>(3 * poly_modulus_degree))));
        BigUInt leading_factor = BigUInt(simulation1.parms_.plain_modulus().bit_count(), *simulation1.parms_.plain_modulus().pointer()) * leading_sqrt_factor;

        BigUInt result_noise = simulation2.noise_ * sqrt_factor_1
            + simulation1.noise_ * sqrt_factor_2
            + sqrt_factor_total;
        result_noise *= leading_factor;

        // If noise overflowed, set it to max
        BigUInt max_noise = simulation1.parms_.coeff_modulus() >> 1;
        if (result_noise > max_noise)
        {
            result_noise = max_noise;
        }
        result_noise.resize(simulation1.parms_.coeff_modulus().bit_count());

        return Simulation(simulation1.parms_, result_ciphertext_size, result_noise);
    }

    Simulation SimulationEvaluator::multiply_plain(const Simulation &simulation, int plain_max_coeff_count, const BigUInt &plain_max_abs_value)
    {
        if (plain_max_coeff_count >= simulation.parms_.poly_modulus().coeff_count() || plain_max_coeff_count <= 0)
        {
            throw invalid_argument("plain_max_coeff_count out of range");
        }

        // Disallow multiply_plain by the zero plaintext
        if (plain_max_abs_value.is_zero() || plain_max_coeff_count == 0)
        {
            throw invalid_argument("plaintext multiplier cannot be zero");
        }

        // Noise is ~ plain_max_coeff_count * plain_max_abs_value * old_noise
        BigUInt result_noise = simulation.noise_ * plain_max_abs_value * static_cast<uint64_t>(plain_max_coeff_count);
        
        return Simulation(simulation.parms_, simulation.ciphertext_size_, result_noise);
    }

    Simulation SimulationEvaluator::multiply_plain(const Simulation &simulation, int plain_max_coeff_count, uint64_t plain_max_abs_value)
    {
        BigUInt plain_max_abs_value_uint;
        plain_max_abs_value_uint = plain_max_abs_value;
        return multiply_plain(simulation, plain_max_coeff_count, plain_max_abs_value_uint);
    }

    Simulation SimulationEvaluator::add_plain(const Simulation &simulation, int plain_max_coeff_count, const BigUInt &plain_max_abs_value)
    {
        if (plain_max_coeff_count >= simulation.parms_.poly_modulus().coeff_count() || plain_max_coeff_count <= 0)
        {
            throw invalid_argument("plain_max_coeff_count out of range");
        }

        int coeff_bit_count = simulation.parms_.coeff_modulus().bit_count();
        int coeff_uint64_count = divide_round_up(coeff_bit_count, bits_per_uint64);

        // Noise is old_noise + r_t(q) * plain_max_coeff_count * plain_max_abs_value
        BigUInt result_noise(coeff_bit_count);

        // Widen plain_modulus_
        ConstPointer wide_plain_modulus = duplicate_uint_if_needed(simulation.parms_.plain_modulus().pointer(), simulation.parms_.plain_modulus().uint64_count(), coeff_uint64_count, false, pool_);

        // Compute summand
        Pointer quotient(allocate_uint(coeff_uint64_count, pool_));
        divide_uint_uint(simulation.parms_.coeff_modulus().pointer(), wide_plain_modulus.get(), coeff_uint64_count, quotient.get(), result_noise.pointer(), pool_);
        result_noise *= plain_max_abs_value * static_cast<uint64_t>(plain_max_coeff_count);

        result_noise += simulation.noise_;

        return Simulation(simulation.parms_, simulation.ciphertext_size_, result_noise);
    }

    Simulation SimulationEvaluator::add_plain(const Simulation &simulation, int plain_max_coeff_count, uint64_t plain_max_abs_value)
    {
        BigUInt plain_max_abs_value_uint;
        plain_max_abs_value_uint = plain_max_abs_value;
        return add_plain(simulation, plain_max_coeff_count, plain_max_abs_value_uint);
    }

    Simulation SimulationEvaluator::sub_plain(const Simulation &simulation, int plain_max_coeff_count, uint64_t plain_max_abs_value)
    {
        BigUInt plain_max_abs_value_uint;
        plain_max_abs_value_uint = plain_max_abs_value;
        return sub_plain(simulation, plain_max_coeff_count, plain_max_abs_value_uint);
    }

    Simulation SimulationEvaluator::get_fresh(const EncryptionParameters &parms, int plain_max_coeff_count, const BigUInt &plain_max_abs_value)
    {
        // Verify parameters
        if (plain_max_coeff_count <= 0 || plain_max_coeff_count >= parms.poly_modulus().coeff_count())
        {
            throw invalid_argument("plain_max_coeff_count is not in the valid range");
        }
        if (plain_max_abs_value.is_zero())
        {
            plain_max_coeff_count = 1;
        }
#ifdef _DEBUG
        if (plain_max_abs_value > ((*parms.plain_modulus().pointer()) >> 1))
        {
            throw invalid_argument("plain_max_abs_value too large");
        }
#endif
        int coeff_bit_count = parms.coeff_modulus().bit_count();
        int coeff_uint64_count = divide_round_up(coeff_bit_count, bits_per_uint64);
        int poly_modulus_degree = parms.poly_modulus().coeff_count() - 1;

        // Widen plain_modulus_ and noise_
        ConstPointer wide_plain_modulus = duplicate_uint_if_needed(parms.plain_modulus().pointer(), parms.plain_modulus().uint64_count(), coeff_uint64_count, false, pool_);
        BigUInt noise(coeff_bit_count);

        // Noise is ~ r_t(q)*plain_max_abs_value*plain_max_coeff_count + 7 * min(B, 6*sigma)*t*n
        Pointer alloc_anchor(allocate_uint(5 * coeff_uint64_count, pool_));
        uint64_t *coeff_div_plain_modulus = alloc_anchor.get();
        uint64_t *quotient = coeff_div_plain_modulus + coeff_uint64_count;
        uint64_t *temp = quotient + coeff_uint64_count;
        uint64_t *temp_high = temp + coeff_uint64_count;

        // First summand
        multiply_uint_uint64(plain_max_abs_value.pointer(), plain_max_abs_value.uint64_count(), static_cast<uint64_t>(plain_max_coeff_count), coeff_uint64_count, noise.pointer());
        divide_uint_uint(parms.coeff_modulus().pointer(), wide_plain_modulus.get(), coeff_uint64_count, quotient, coeff_div_plain_modulus, pool_);
        multiply_truncate_uint_uint(noise.pointer(), coeff_div_plain_modulus, coeff_uint64_count, temp);

        // Second summand
        multiply_uint_uint64(wide_plain_modulus.get(), coeff_bit_count, 7 * static_cast<uint64_t>(min(parms.noise_max_deviation(), parms.noise_standard_deviation() * 6)) * poly_modulus_degree, coeff_uint64_count, temp_high);

        // Result
        add_uint_uint(temp, temp_high, coeff_uint64_count, noise.pointer());
        return Simulation(parms, 2, noise);
    }

    Simulation SimulationEvaluator::get_fresh(const EncryptionParameters &parms, int plain_max_coeff_count, uint64_t plain_max_abs_value)
    {
        return get_fresh(parms, plain_max_coeff_count, BigUInt(bits_per_uint64, plain_max_abs_value));
    }

    Simulation SimulationEvaluator::multiply_many(vector<Simulation> simulations)
    {
        size_t original_size = simulations.size();

        // Verify parameters.
        if (original_size == 0)
        {
            throw invalid_argument("simulations vector cannot be empty");
        }

        // Return quickly if there is nothing to do.
        if (original_size == 1)
        {
            return simulations[0];
        }

        // Repeatedly multiply and add to the back of the vector until the end is reached.
        // We will detect here if one of the elements of the simulations vector was constructed with
        // different encryption parameters than the others.
        for (size_t i = 0; i < simulations.size() - 1; i += 2)
        {
            simulations.emplace_back(relinearize(multiply(simulations[i], simulations[i + 1])));
        }

        return simulations[simulations.size() - 1];
    }

    Simulation SimulationEvaluator::exponentiate(const Simulation &simulation, uint64_t exponent)
    {
        if (exponent == 0)
        {
            throw invalid_argument("exponent cannot be zero");
        }

        // Fast path
        if (exponent == 1)
        {
            return simulation;
        }

        vector<Simulation> exp_vector(exponent, simulation);
        return multiply_many(exp_vector);

        // Binary exponentiation
        /*
        if (exponent % 2 == 0)
        {
            return exponentiate(multiply(simulation, simulation), exponent >> 1);
        }
        return multiply(exponentiate(multiply(simulation, simulation), (exponent - 1) >> 1), simulation);
        */
    }
}
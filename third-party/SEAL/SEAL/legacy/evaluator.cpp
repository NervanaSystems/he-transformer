#include <algorithm>
#include <stdexcept>
#include "evaluator.h"
#include "util/common.h"
#include "util/uintcore.h"
#include "util/uintarith.h"
#include "util/uintarithmod.h"
#include "util/polycore.h"
#include "util/polyarith.h"
#include "util/polyarithmod.h"
#include "util/nussbaumer.h"
#include "util/polyfftmultmod.h"
#include "bigpoly.h"
#include "util/ntt.h"

using namespace std;
using namespace seal::util;

namespace seal
{
    namespace
    {
        ConstPointer duplicate_poly_if_needed(const BigPoly &poly, int new_coeff_count, int new_coeff_uint64_count, bool force, MemoryPool &pool_)
        {
            return duplicate_poly_if_needed(poly.pointer(), poly.coeff_count(), poly.coeff_uint64_count(), 
                new_coeff_count, new_coeff_uint64_count, force, pool_);
        }

        ConstPointer duplicate_poly_if_needed(const BigPoly &poly, bool force, MemoryPool &pool_)
        {
            return duplicate_if_needed(poly.pointer(), poly.coeff_count() * poly.coeff_uint64_count(), force, pool_);
        }

        ConstPointer duplicate_bigpolyarray_if_needed(const BigPolyArray &operand, int new_size, int new_coeff_count, int new_coeff_uint64_count, bool force, MemoryPool &pool_)
        {
            return duplicate_bigpolyarray_if_needed(operand.pointer(0), operand.size(), operand.coeff_count(), operand.coeff_uint64_count(), 
                new_size, new_coeff_count, new_coeff_uint64_count, force, pool_);
        }

        ConstPointer duplicate_bigpolyarray_if_needed(const BigPolyArray &operand, bool force, MemoryPool &pool_)
        {
            return duplicate_if_needed(operand.pointer(0), operand.size() * operand.coeff_count() * operand.coeff_uint64_count(), force, pool_);
        }

        bool are_poly_coefficients_less_than(const BigPoly &poly, const BigUInt &max_coeff)
        {
            return util::are_poly_coefficients_less_than(poly.pointer(), poly.coeff_count(), poly.coeff_uint64_count(), max_coeff.pointer(), max_coeff.uint64_count());
        }
    }

    Evaluator::Evaluator(const Evaluator &copy) : 
        pool_(copy.pool_), parms_(copy.parms_), qualifiers_(copy.qualifiers_),
        ntt_tables_(copy.ntt_tables_), evaluation_keys_(copy.evaluation_keys_),
        product_coeff_bit_count_(copy.product_coeff_bit_count_)
    {
        int coeff_count = parms_.poly_modulus().coeff_count();
        int poly_coeff_uint64_count = parms_.poly_modulus().coeff_uint64_count();
        int coeff_uint64_count = parms_.coeff_modulus().uint64_count();

        // Allocate memory and copy over values
        upper_half_threshold_ = allocate_uint(coeff_uint64_count, pool_);
        set_uint_uint(copy.upper_half_threshold_.get(), coeff_uint64_count, upper_half_threshold_.get());
        upper_half_increment_ = allocate_uint(coeff_uint64_count, pool_);
        set_uint_uint(copy.upper_half_increment_.get(), coeff_uint64_count, upper_half_increment_.get());

        plain_upper_half_threshold_ = allocate_uint(coeff_uint64_count, pool_);
        set_uint_uint(copy.plain_upper_half_threshold_.get(), coeff_uint64_count, plain_upper_half_threshold_.get());

        plain_upper_half_increment_ = allocate_uint(coeff_uint64_count, pool_);
        set_uint_uint(copy.plain_upper_half_increment_.get(), coeff_uint64_count, plain_upper_half_increment_.get());

        coeff_div_plain_modulus_ = allocate_uint(coeff_uint64_count, pool_);
        set_uint_uint(copy.coeff_div_plain_modulus_.get(), coeff_uint64_count, coeff_div_plain_modulus_.get());

        coeff_modulus_div_two_ = allocate_uint(coeff_uint64_count, pool_);
        set_uint_uint(copy.coeff_modulus_div_two_.get(), coeff_uint64_count, coeff_modulus_div_two_.get());

        coeff_div_plain_modulus_ = allocate_uint(coeff_uint64_count, pool_);
        set_uint_uint(copy.coeff_div_plain_modulus_.get(), coeff_uint64_count, coeff_div_plain_modulus_.get());

        // Initialize moduli.
        polymod_ = PolyModulus(parms_.poly_modulus().pointer(), coeff_count, poly_coeff_uint64_count);
        mod_ = Modulus(parms_.coeff_modulus().pointer(), coeff_uint64_count, pool_);
    }

    void Evaluator::negate(const Ciphertext &encrypted, Ciphertext &destination)
    {
        const BigPolyArray &encrypted_array = encrypted.get_array();
        BigPolyArray &destination_array = destination.get_mutable_array();

        // Extract encryption parameters.
        int coeff_count = parms_.poly_modulus().coeff_count();
        int coeff_bit_count = parms_.coeff_modulus().bit_count();
        int coeff_uint64_count = parms_.coeff_modulus().uint64_count();
        int count = encrypted_array.size();

        // Verify parameters.
        if (encrypted.hash_block_ != parms_.get_hash_block())
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }

        ConstPointer encrypted_safe = duplicate_bigpolyarray_if_needed(encrypted_array, &encrypted == &destination, pool_);

        // Prepare destination
        destination_array.resize(count, coeff_count, coeff_bit_count);
        destination.hash_block_ = parms_.get_hash_block();

        // Negate each bigpoly in the array
        negate_poly_coeffmod(encrypted_safe.get(), coeff_count * count, parms_.coeff_modulus().pointer(), coeff_uint64_count, destination_array.pointer(0));
    }

    void Evaluator::add(const Ciphertext &encrypted1, const Ciphertext &encrypted2, Ciphertext &destination)
    {
        const BigPolyArray &encrypted1_array = encrypted1.get_array();
        const BigPolyArray &encrypted2_array = encrypted2.get_array();
        BigPolyArray &destination_array = destination.get_mutable_array();

        // Extract encryption parameters.
        int coeff_count = parms_.poly_modulus().coeff_count();
        int coeff_bit_count = parms_.coeff_modulus().bit_count();
        int coeff_uint64_count = parms_.coeff_modulus().uint64_count();
        int encrypted1_count = encrypted1_array.size();
        int encrypted2_count = encrypted2_array.size();
        int max_count = max(encrypted1_count, encrypted2_count);
        int min_count = min(encrypted1_count, encrypted2_count);

        // Verify parameters.
        if (encrypted1.hash_block_ != parms_.get_hash_block())
        {
            throw invalid_argument("encrypted1 is not valid for encryption parameters");
        }
        if (encrypted2.hash_block_ != parms_.get_hash_block())
        {
            throw invalid_argument("encrypted2 is not valid for encryption parameters");
        }

        ConstPointer encrypted1_safe = duplicate_bigpolyarray_if_needed(encrypted1_array, &encrypted1 == &destination, pool_);
        ConstPointer encrypted2_safe = duplicate_bigpolyarray_if_needed(encrypted2_array, &encrypted2 == &destination, pool_);

        // Prepare destination
        destination_array.resize(max_count, coeff_count, coeff_bit_count);
        destination.hash_block_ = parms_.get_hash_block();

        // Add BigPolyArrays
        add_poly_poly_coeffmod(encrypted1_safe.get(), encrypted2_safe.get(), coeff_count * min_count, parms_.coeff_modulus().pointer(), coeff_uint64_count, destination_array.pointer(0));

        // If the sizes are the same, we are done
        if (max_count == min_count)
        {
            return;
        }

        // Copy the remainding BigPolys of the array with larger count into destination
        set_poly_poly((encrypted1_count == max_count ? encrypted1_safe.get() : encrypted2_safe.get()) + min_count * coeff_count * coeff_uint64_count,
            coeff_count * (max_count - min_count), coeff_uint64_count, destination_array.pointer(min_count));
    }

    void Evaluator::add_many(const vector<Ciphertext> &encrypteds, Ciphertext &destination)
    {
        if (encrypteds.empty())
        {
            throw invalid_argument("encrypteds cannot be empty");
        }

        destination = encrypteds[0];
        for (size_t i = 1; i < encrypteds.size(); i++)
        {
            add(destination, encrypteds[i], destination);
        }
    }

    void Evaluator::sub(const Ciphertext &encrypted1, const Ciphertext &encrypted2, Ciphertext &destination)
    {
        const BigPolyArray &encrypted1_array = encrypted1.get_array();
        const BigPolyArray &encrypted2_array = encrypted2.get_array();
        BigPolyArray &destination_array = destination.get_mutable_array();

        // Extract encryption parameters.
        int coeff_count = parms_.poly_modulus().coeff_count();
        int coeff_bit_count = parms_.coeff_modulus().bit_count();
        int coeff_uint64_count = parms_.coeff_modulus().uint64_count();
        int encrypted1_count = encrypted1_array.size();
        int encrypted2_count = encrypted2_array.size();
        int max_count = max(encrypted1_count, encrypted2_count);
        int min_count = min(encrypted1_count, encrypted2_count);

        // Verify parameters.
        if (encrypted1.hash_block_ != parms_.get_hash_block())
        {
            throw invalid_argument("encrypted1 is not valid for encryption parameters");
        }
        if (encrypted2.hash_block_ != parms_.get_hash_block())
        {
            throw invalid_argument("encrypted2 is not valid for encryption parameters");
        }

        ConstPointer encrypted1_safe = duplicate_bigpolyarray_if_needed(encrypted1_array, &encrypted1 == &destination, pool_);
        ConstPointer encrypted2_safe = duplicate_bigpolyarray_if_needed(encrypted2_array, &encrypted2 == &destination, pool_);

        // Prepare destination
        destination_array.resize(max_count, coeff_count, coeff_bit_count);
        destination.hash_block_ = parms_.get_hash_block();

        // Subtract polynomials.
        sub_poly_poly_coeffmod(encrypted1_safe.get(), encrypted2_safe.get(), coeff_count * min_count, parms_.coeff_modulus().pointer(), coeff_uint64_count, destination_array.pointer(0));

        // If the sizes are the same, we are done
        if (max_count == min_count)
        {
            return;
        }

        // If encrypted1 has larger count, copy remaining entries into destination_array.
        if (encrypted1_count == max_count) 
        {
            // Copy the remainding BigPolys of the array with larger count into destination
            set_poly_poly(encrypted1_safe.get() + min_count * coeff_count * coeff_uint64_count, coeff_count * (max_count - min_count), coeff_uint64_count, destination_array.pointer(min_count));
        }
        // Otherwise, encrypted2 has larger count, negate remaining entries and copy into destination_array.
        else
        {
            negate_poly_coeffmod(encrypted2_safe.get() + min_count * coeff_count * coeff_uint64_count, coeff_count * (max_count - min_count), parms_.coeff_modulus().pointer(), coeff_uint64_count, destination_array.pointer(min_count));
        }
    }

    void Evaluator::multiply(const Ciphertext &encrypted1, const Ciphertext &encrypted2, Ciphertext &destination)
    {
        const BigPolyArray &encrypted1_array = encrypted1.get_array();
        const BigPolyArray &encrypted2_array = encrypted2.get_array();
        BigPolyArray &destination_array = destination.get_mutable_array();

        // Extract encryption parameters.
        int coeff_count = parms_.poly_modulus().coeff_count();
        int coeff_bit_count = parms_.coeff_modulus().bit_count();
        int coeff_uint64_count = parms_.coeff_modulus().uint64_count();
        int encrypted1_count = encrypted1_array.size();
        int encrypted2_count = encrypted2_array.size();
        int encrypted_ptr_increment = coeff_count * coeff_uint64_count;

        // Determine destination_array.size()
        int dest_count = encrypted1_count + encrypted2_count - 1; // default is 3 (c_0, c_1, c_2)

        // Verify parameters.
        if (encrypted1.hash_block_ != parms_.get_hash_block())
        {
            throw invalid_argument("encrypted1 is not valid for encryption parameters");
        }
        if (encrypted2.hash_block_ != parms_.get_hash_block())
        {
            throw invalid_argument("encrypted2 is not valid for encryption parameters");
        }

        ConstPointer encrypted1_safe = duplicate_bigpolyarray_if_needed(encrypted1_array, &encrypted1 == &destination, pool_);
        ConstPointer encrypted2_safe = duplicate_bigpolyarray_if_needed(encrypted2_array, &encrypted2 == &destination, pool_);

        // Prepare destination
        destination_array.resize(dest_count, coeff_count, coeff_bit_count);
        destination.hash_block_ = parms_.get_hash_block();
        destination_array.set_zero();

        if (!qualifiers_.enable_ntt_in_multiply && qualifiers_.enable_fft)
        {
            // Significant size of plain modulus
            int plain_bit_count = parms_.plain_modulus().bit_count();
            int plain_uint64_count = parms_.plain_modulus().uint64_count();

            // Store the intermediate sizes until do_fft_base is reached (nq). 
            int sum_bit_count = get_significant_bit_count(static_cast<uint64_t>(coeff_count)) + coeff_bit_count + 1; 
            int sum_uint64_count = divide_round_up(sum_bit_count, bits_per_uint64); 

            // Stores sums of products of polynomials
            int wide_product_bit_count = product_coeff_bit_count_
                + min(get_significant_bit_count(static_cast<uint64_t>(encrypted1_count)), get_significant_bit_count(static_cast<uint64_t>(encrypted2_count)));
            int wide_product_uint64_count = divide_round_up(wide_product_bit_count, bits_per_uint64);

            // Stores the sums multiplied by plain modulus
            int intermediate_bit_count = wide_product_bit_count + plain_bit_count;
            int intermediate_uint64_count = divide_round_up(intermediate_bit_count, bits_per_uint64);

            Pointer intermediate(allocate_uint(intermediate_uint64_count, pool_));
            Pointer quotient(allocate_uint(intermediate_uint64_count, pool_));

            if (encrypted1_count == 2 && encrypted2_count == 2)
            {
                // Fast logic for size 2 ciphertexts

                // It is very important to set the leading coefficients to zero here, as the fft multiplication functions are not touching it 
                // (they compute the number of terms as 1 << coeff_count_power, which is one less than the true coefficient count.
                Pointer temp(allocate_poly(coeff_count, wide_product_uint64_count, pool_));
                Pointer temp2(allocate_poly(coeff_count, wide_product_uint64_count, pool_));
                int last_coeff_shift = (coeff_count - 1) * wide_product_uint64_count;
                set_zero_uint(wide_product_uint64_count, temp.get() + last_coeff_shift);
                set_zero_uint(wide_product_uint64_count, temp2.get() + last_coeff_shift);

                // Increase uint64 count by one to be able to always accommodate sums of coefficients.
                // This is especially important for coeff_moduli of form 2^k-1.
                // int sum_coeff_uint64_count = coeff_uint64_count + 1;
                Pointer temp3(allocate_poly(coeff_count, sum_uint64_count, pool_));
                Pointer temp4(allocate_poly(coeff_count, sum_uint64_count, pool_));
                Pointer temp5(allocate_poly(coeff_count, sum_uint64_count, pool_));

                Pointer temp6(allocate_poly(coeff_count, wide_product_uint64_count, pool_));
                set_zero_uint(wide_product_uint64_count, temp6.get() + last_coeff_shift);

                nussbaumer_multiply_poly_poly(encrypted1_safe.get(), encrypted2_safe.get(), polymod_.coeff_count_power_of_two(), coeff_uint64_count, sum_uint64_count, wide_product_uint64_count, temp.get(), pool_);
                nussbaumer_multiply_poly_poly(encrypted1_safe.get() + encrypted_ptr_increment, encrypted2_safe.get() + encrypted_ptr_increment, polymod_.coeff_count_power_of_two(), coeff_uint64_count, sum_uint64_count, wide_product_uint64_count, temp2.get(), pool_);
                
                set_poly_poly(encrypted1_safe.get(), coeff_count, coeff_uint64_count, coeff_count, sum_uint64_count, temp3.get());
                set_poly_poly(encrypted1_safe.get() + encrypted_ptr_increment, coeff_count, coeff_uint64_count, coeff_count, sum_uint64_count, temp4.get());
                add_poly_poly(temp3.get(), temp4.get(), coeff_count, sum_uint64_count, temp3.get());

                set_poly_poly(encrypted2_safe.get(), coeff_count, coeff_uint64_count, coeff_count, sum_uint64_count, temp4.get());
                set_poly_poly(encrypted2_safe.get() + encrypted_ptr_increment, coeff_count, coeff_uint64_count, coeff_count, sum_uint64_count, temp5.get());
                add_poly_poly(temp4.get(), temp5.get(), coeff_count, sum_uint64_count, temp4.get());

                nussbaumer_multiply_poly_poly(temp3.get(), temp4.get(), polymod_.coeff_count_power_of_two(), sum_uint64_count, sum_uint64_count, wide_product_uint64_count, temp6.get(), pool_);
                
                sub_poly_poly(temp6.get(), temp.get(), coeff_count, wide_product_uint64_count, temp6.get());
                sub_poly_poly(temp6.get(), temp2.get(), coeff_count, wide_product_uint64_count, temp6.get());

                uint64_t *temp_dest_coeff0 = temp.get();
                uint64_t *dest_coeff0 = destination_array.pointer(0);
                uint64_t *temp_dest_coeff1 = temp6.get();
                uint64_t *dest_coeff1 = destination_array.pointer(1);
                uint64_t *temp_dest_coeff2 = temp2.get();
                uint64_t *dest_coeff2 = destination_array.pointer(2);

                for (int coeff_index = 0; coeff_index < coeff_count; coeff_index++)
                {
                    // Dest[0]
                    bool coeff_is_negative = is_high_bit_set_uint(temp_dest_coeff0, wide_product_uint64_count);
                    if (coeff_is_negative)
                    {
                        negate_uint(temp_dest_coeff0, wide_product_uint64_count, temp_dest_coeff0);
                    }

                    multiply_uint_uint(temp_dest_coeff0, wide_product_uint64_count, parms_.plain_modulus().pointer(), plain_uint64_count, intermediate_uint64_count, intermediate.get());
                    add_uint_uint(intermediate.get(), intermediate_uint64_count, coeff_modulus_div_two_.get(), coeff_uint64_count, false, intermediate_uint64_count, intermediate.get());
                    divide_uint_uint_inplace(intermediate.get(), mod_, intermediate_uint64_count, quotient.get(), pool_);
                    modulo_uint_inplace(quotient.get(), intermediate_uint64_count, mod_, pool_);
                
                    if (coeff_is_negative)
                    {
                        negate_uint_mod(quotient.get(), parms_.coeff_modulus().pointer(), coeff_uint64_count, dest_coeff0);
                    }
                    else
                    {
                        set_uint_uint(quotient.get(), coeff_uint64_count, dest_coeff0);
                    }

                    temp_dest_coeff0 += wide_product_uint64_count;
                    dest_coeff0 += coeff_uint64_count;
                
                    // Dest[1]
                    coeff_is_negative = is_high_bit_set_uint(temp_dest_coeff1, wide_product_uint64_count);
                    if (coeff_is_negative)
                    {
                        negate_uint(temp_dest_coeff1, wide_product_uint64_count, temp_dest_coeff1);
                    }
                
                    multiply_uint_uint(temp_dest_coeff1, wide_product_uint64_count, parms_.plain_modulus().pointer(), plain_uint64_count, intermediate_uint64_count, intermediate.get());
                    add_uint_uint(intermediate.get(), intermediate_uint64_count, coeff_modulus_div_two_.get(), coeff_uint64_count, false, intermediate_uint64_count, intermediate.get());
                    divide_uint_uint_inplace(intermediate.get(), mod_, intermediate_uint64_count, quotient.get(), pool_);
                    modulo_uint_inplace(quotient.get(), intermediate_uint64_count, mod_, pool_);
                
                    if (coeff_is_negative)
                    {
                        negate_uint_mod(quotient.get(), parms_.coeff_modulus().pointer(), coeff_uint64_count, dest_coeff1);
                    }
                    else
                    {
                        set_uint_uint(quotient.get(), coeff_uint64_count, dest_coeff1);
                    }

                    temp_dest_coeff1 += wide_product_uint64_count;
                    dest_coeff1 += coeff_uint64_count;
                
                    // Dest[2]
                    coeff_is_negative = is_high_bit_set_uint(temp_dest_coeff2, wide_product_uint64_count);
                    if (coeff_is_negative)
                    {
                        negate_uint(temp_dest_coeff2, wide_product_uint64_count, temp_dest_coeff2);
                    }

                    multiply_uint_uint(temp_dest_coeff2, wide_product_uint64_count, parms_.plain_modulus().pointer(), plain_uint64_count, intermediate_uint64_count, intermediate.get());
                    add_uint_uint(intermediate.get(), intermediate_uint64_count, coeff_modulus_div_two_.get(), coeff_uint64_count, false, intermediate_uint64_count, intermediate.get());
                    divide_uint_uint_inplace(intermediate.get(), mod_, intermediate_uint64_count, quotient.get(), pool_);
                    modulo_uint_inplace(quotient.get(), intermediate_uint64_count, mod_, pool_);
                
                    if (coeff_is_negative)
                    {
                        negate_uint_mod(quotient.get(), parms_.coeff_modulus().pointer(), coeff_uint64_count, dest_coeff2);
                    }
                    else
                    {
                        set_uint_uint(quotient.get(), coeff_uint64_count, dest_coeff2);
                    }

                    temp_dest_coeff2 += wide_product_uint64_count;
                    dest_coeff2 += coeff_uint64_count;
                }

                return;
            }
            else
            {
                // Arbitrary size ciphertexts

                // To store the current product of individual polynomials
                Pointer temp_poly(allocate_poly(coeff_count, wide_product_uint64_count, pool_));

                // To store the current component of dest before scaling or modular reduction
                Pointer temp_dest(allocate_poly(coeff_count, wide_product_uint64_count, pool_));

                // each index of destination corresponds to the secret key s^index
                // loop over temp_dest, filling in the appropriate cross terms for each index,
                // to be scaled and modular reduced later into destination for each index
                int current_encrypted1_limit = 0;
                for (int secret_power_index = 0; secret_power_index < dest_count; ++secret_power_index)
                {
                    // Reset temp_dest
                    set_zero_poly(coeff_count, wide_product_uint64_count, temp_dest.get());

                    // loop over encrypted1 components [i], seeing if a match exists with an encrypted2 component [j] such that [i+j]=[secret_power_index]
                    // only need to check encrypted1 components up to and including [secret_power_index], and strictly less than [encrypted_array.size()]
                    current_encrypted1_limit = min(encrypted1_count, secret_power_index + 1);

                    for (int encrypted1_index = 0; encrypted1_index < current_encrypted1_limit; ++encrypted1_index)
                    {
                        // check if a corresponding component in encrypted2 exists
                        if (encrypted2_count > secret_power_index - encrypted1_index)
                        {
                            int encrypted2_index = secret_power_index - encrypted1_index;

                            // reset temp_poly
                            set_zero_poly(coeff_count, wide_product_uint64_count, temp_poly.get());

                            // fft multiply encrypted1_array[encrypted1_index] with encrypted2_array[encrypted2_index] and store in temp_poly
                            nussbaumer_multiply_poly_poly(encrypted1_safe.get() + encrypted_ptr_increment * encrypted1_index, encrypted2_safe.get() + encrypted_ptr_increment * encrypted2_index, polymod_.coeff_count_power_of_two(), coeff_uint64_count, sum_uint64_count, wide_product_uint64_count, temp_poly.get(), pool_);

                            // update temp_dest with current value of temp_poly
                            add_poly_poly(temp_poly.get(), temp_dest.get(), coeff_count, wide_product_uint64_count, temp_dest.get());
                        }
                    }

                    // when temp_dest is fully populated,  multiply by plain_modulus and divide by coeff_modulus and then modulo by coeff_modulus
                    // put the result in  destination_array[secret_power_index]
                    uint64_t *temp_dest_coeff = temp_dest.get();
                    uint64_t *dest_coeff = destination_array.pointer(secret_power_index);

                    for (int coeff_index = 0; coeff_index < coeff_count; coeff_index++)
                    {
                        bool coeff_is_negative = is_high_bit_set_uint(temp_dest_coeff, wide_product_uint64_count);
                        if (coeff_is_negative)
                        {
                            negate_uint(temp_dest_coeff, wide_product_uint64_count, temp_dest_coeff);
                        }

                        multiply_uint_uint(temp_dest_coeff, wide_product_uint64_count, parms_.plain_modulus().pointer(), plain_uint64_count, intermediate_uint64_count, intermediate.get());
                        add_uint_uint(intermediate.get(), intermediate_uint64_count, coeff_modulus_div_two_.get(), coeff_uint64_count, false, intermediate_uint64_count, intermediate.get());
                        divide_uint_uint_inplace(intermediate.get(), mod_, intermediate_uint64_count, quotient.get(), pool_);
                        modulo_uint_inplace(quotient.get(), intermediate_uint64_count, mod_, pool_);

                        if (coeff_is_negative)
                        {
                            negate_uint_mod(quotient.get(), parms_.coeff_modulus().pointer(), coeff_uint64_count, dest_coeff);
                        }
                        else
                        {
                            set_uint_uint(quotient.get(), coeff_uint64_count, dest_coeff);
                        }

                        temp_dest_coeff += wide_product_uint64_count;
                        dest_coeff += coeff_uint64_count;
                    }
                }

                return;
            }
        }
        else
        {
            // This branch should never be reached
            throw logic_error("invalid encryption parameters");

            //// NON-NTT/NON-NUSSBAUMER PARAMETERS ARE NOT SUPPORTED

            //// Size of the product of polynomials
            //int product_coeff_count = 2 * coeff_count - 1;

            //// Significant size of plain modulus
            //int plain_uint64_count = parms_.plain_modulus().uint64_count();

            //// Stores sums of products of polynomials
            //int wide_product_bit_count = product_coeff_bit_count_
            //    + min(get_significant_bit_count(static_cast<uint64_t>(encrypted1_count)), get_significant_bit_count(static_cast<uint64_t>(encrypted2_count)));
            //int wide_product_uint64_count = divide_round_up(wide_product_bit_count, bits_per_uint64);

            //// Stores the sums multiplied by plain modulus
            //int intermediate_bit_count = wide_product_bit_count + plain_bit_count;
            //int intermediate_uint64_count = divide_round_up(intermediate_bit_count, bits_per_uint64);

            //Pointer intermediate(allocate_uint(intermediate_uint64_count, pool_));
            //Pointer quotient(allocate_uint(intermediate_uint64_count, pool_));

            //// To store the current product of individual polynomials
            //Pointer temp_poly(allocate_poly(product_coeff_count, wide_product_uint64_count, pool_));

            //// To store the current component of dest before scaling or any reduction
            //Pointer temp_dest(allocate_poly(product_coeff_count, wide_product_uint64_count, pool_));

            //// To store result of temp_dest reduced modulo coeffmod
            //Pointer temp_dest_mod(allocate_poly(product_coeff_count, coeff_uint64_count, pool_));

            //// each index of destination corresponds to the secret key s^index
            //// loop over dest_temp, filling in the appropriate cross terms for each index, to be scaled and modular reduced later into destination
            //// for each index, we will loop over indices of encrypted1
            //// initialize a variable to denote when to stop looping of encrypted1
            //int current_encrypted1_limit = 0;

            //for (int secret_power_index = 0; secret_power_index < dest_count; ++secret_power_index)
            //{
            //    // reset temp_dest and temp_dest_mod
            //    set_zero_poly(product_coeff_count, wide_product_uint64_count, temp_dest.get());
            //    set_zero_poly(product_coeff_count, coeff_uint64_count, temp_dest_mod.get());

            //    // loop over encrypted1 components [i], seeing if a match exists with an encrypted2 component [j] such that [i+j]=[secret_power_index]
            //    // only need to check encrypted1 components up to and including [secret_power_index], and strictly less than [encrypted_array.size()]
            //    current_encrypted1_limit = min(encrypted1_count, secret_power_index + 1);

            //    for (int encrypted1_index = 0; encrypted1_index < current_encrypted1_limit; ++encrypted1_index)
            //    {
            //        // check if a corresponding component in encrypted2 exists
            //        if (encrypted2_count > secret_power_index - encrypted1_index)
            //        {
            //            int encrypted2_index = secret_power_index - encrypted1_index;

            //            // reset temp_poly
            //            set_zero_poly(product_coeff_count, wide_product_uint64_count, temp_poly.get());

            //            //multiply encrypted1_array[encrypted1_index] with encrypted2_array[encrypted2_index] and store in temp_poly
            //            multiply_poly_poly(encrypted1_safe.get() + encrypted_ptr_increment * encrypted1_index, coeff_count, coeff_uint64_count,
            //                encrypted2_safe.get() + encrypted_ptr_increment * encrypted2_index, coeff_count, coeff_uint64_count,
            //                product_coeff_count, wide_product_uint64_count, temp_poly.get(), pool_);

            //            // update temp_dest with current value of temp_poly
            //            add_poly_poly(temp_poly.get(), temp_dest.get(), product_coeff_count, wide_product_uint64_count, temp_dest.get());
            //        }
            //    }
            //    // when temp_dest is fully populated,  multiply by plain_modulus and divide by coeff_modulus and then modulo by coeff_modulus
            //    // put result into temp_dest_mod
            //    uint64_t *temp_dest_coeff = temp_dest.get();
            //    uint64_t *temp_dest_mod_coeff = temp_dest_mod.get();

            //    for (int coeff_index = 0; coeff_index < product_coeff_count; coeff_index++)
            //    {
            //        multiply_uint_uint(temp_dest_coeff, wide_product_uint64_count, parms_.plain_modulus().pointer(), plain_uint64_count, intermediate_uint64_count, intermediate.get());
            //        add_uint_uint(intermediate.get(), intermediate_uint64_count, coeff_modulus_div_two_.pointer(), coeff_uint64_count, false, intermediate_uint64_count, intermediate.get());
            //        divide_uint_uint_inplace(intermediate.get(), mod_, intermediate_uint64_count, quotient.get(), pool_);
            //        modulo_uint_inplace(quotient.get(), intermediate_uint64_count, mod_, pool_);

            //        set_uint_uint(quotient.get(), coeff_uint64_count, temp_dest_mod_coeff);

            //        temp_dest_coeff += wide_product_uint64_count;
            //        temp_dest_mod_coeff += coeff_uint64_count;
            //    }

            //    // Perform polynomial modulo.
            //    modulo_poly_inplace(temp_dest_mod.get(), product_coeff_count, polymod_, mod_, pool_);

            //    // Copy to destination_array.
            //    set_poly_poly(temp_dest_mod.get(), coeff_count, coeff_uint64_count, destination_array.pointer(secret_power_index));
            //}
        }
    }

    void Evaluator::square(const Ciphertext &encrypted, Ciphertext &destination)
    {
        const BigPolyArray &encrypted_array = encrypted.get_array();
        BigPolyArray &destination_array = destination.get_mutable_array();

        int encrypted_count = encrypted_array.size();

        // Optimization implemented currently only for size 2 ciphertexts
        if (encrypted_count != 2)
        {
            multiply(encrypted, encrypted, destination);
            return;
        }

        int coeff_count = parms_.poly_modulus().coeff_count();
        int coeff_bit_count = parms_.coeff_modulus().bit_count();
        int coeff_uint64_count = parms_.coeff_modulus().uint64_count();
        int encrypted_ptr_increment = coeff_count * coeff_uint64_count;

        // Determine destination_array.size()
        int dest_count = (encrypted_count << 1) - 1;

        // Verify parameters.
        if (encrypted.hash_block_ != parms_.get_hash_block())
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }

        ConstPointer encrypted_safe = duplicate_bigpolyarray_if_needed(encrypted_array, &encrypted == &destination, pool_);

        // Prepare destination
        destination_array.resize(dest_count, coeff_count, coeff_bit_count);
        destination.hash_block_ = parms_.get_hash_block();
        destination_array.set_zero();

        if (!qualifiers_.enable_ntt_in_multiply && qualifiers_.enable_fft)
        {
            // Use Nussbaumer convolution for all multiplications

            // Significant size of plain modulus
            int plain_bit_count = parms_.plain_modulus().bit_count();
            int plain_uint64_count = parms_.plain_modulus().uint64_count();

            // Stores sums of products of polynomials
            int wide_product_bit_count = product_coeff_bit_count_ + get_significant_bit_count(static_cast<uint64_t>(encrypted_count << 1));
            int wide_product_uint64_count = divide_round_up(wide_product_bit_count, bits_per_uint64);
            int sum_bit_count = coeff_bit_count + plain_bit_count + 1;
            int sum_uint64_count = divide_round_up(sum_bit_count, bits_per_uint64);

            // Stores the sums multiplied by plain modulus
            int intermediate_bit_count = wide_product_bit_count + plain_bit_count;
            int intermediate_uint64_count = divide_round_up(intermediate_bit_count, bits_per_uint64);

            Pointer intermediate(allocate_uint(intermediate_uint64_count, pool_));
            Pointer quotient(allocate_uint(intermediate_uint64_count, pool_));

            // It is very important to set the leading coefficients to zero here, as the fft multiplication functions are not touching it 
            // (they compute the number of terms as 1 << coeff_count_power, which is one less than the true coefficient count.
            Pointer temp1(allocate_poly(coeff_count, wide_product_uint64_count, pool_)); // c0^2
            Pointer temp2(allocate_poly(coeff_count, wide_product_uint64_count, pool_)); // c1^2
            Pointer temp3(allocate_poly(coeff_count, wide_product_uint64_count, pool_)); // c0*c1
            int last_coeff_shift = (coeff_count - 1) * wide_product_uint64_count;
            set_zero_uint(wide_product_uint64_count, temp1.get() + last_coeff_shift);
            set_zero_uint(wide_product_uint64_count, temp2.get() + last_coeff_shift);
            set_zero_uint(wide_product_uint64_count, temp3.get() + last_coeff_shift);

            nussbaumer_cross_multiply_poly_poly(encrypted_safe.get(), encrypted_safe.get() + encrypted_ptr_increment, polymod_.coeff_count_power_of_two(), coeff_uint64_count, sum_uint64_count, wide_product_uint64_count, temp1.get(), temp2.get(), temp3.get(), pool_);

            uint64_t *temp_dest_coeff0 = temp1.get();
            uint64_t *dest_coeff0 = destination_array.pointer(0);
            uint64_t *temp_dest_coeff1 = temp3.get();
            uint64_t *dest_coeff1 = destination_array.pointer(1);
            uint64_t *temp_dest_coeff2 = temp2.get();
            uint64_t *dest_coeff2 = destination_array.pointer(2);

            for (int coeff_index = 0; coeff_index < coeff_count; coeff_index++)
            {
                // Dest[0]
                bool coeff_is_negative = is_high_bit_set_uint(temp_dest_coeff0, wide_product_uint64_count);
                if (coeff_is_negative)
                {
                    negate_uint(temp_dest_coeff0, wide_product_uint64_count, temp_dest_coeff0);
                }

                multiply_uint_uint(temp_dest_coeff0, wide_product_uint64_count, parms_.plain_modulus().pointer(), plain_uint64_count, intermediate_uint64_count, intermediate.get());
                add_uint_uint(intermediate.get(), intermediate_uint64_count, coeff_modulus_div_two_.get(), coeff_uint64_count, false, intermediate_uint64_count, intermediate.get());
                divide_uint_uint_inplace(intermediate.get(), mod_, intermediate_uint64_count, quotient.get(), pool_);
                modulo_uint_inplace(quotient.get(), intermediate_uint64_count, mod_, pool_);

                if (coeff_is_negative)
                {
                    negate_uint_mod(quotient.get(), parms_.coeff_modulus().pointer(), coeff_uint64_count, dest_coeff0);
                }
                else
                {
                    set_uint_uint(quotient.get(), coeff_uint64_count, dest_coeff0);
                }

                temp_dest_coeff0 += wide_product_uint64_count;
                dest_coeff0 += coeff_uint64_count;

                // Dest[1] : c0*c1 -> 2c0*c1
                coeff_is_negative = is_high_bit_set_uint(temp_dest_coeff1, wide_product_uint64_count);
                if (coeff_is_negative)
                {
                    negate_uint(temp_dest_coeff1, wide_product_uint64_count, temp_dest_coeff1);
                }

                // Multiply by 2
                left_shift_uint(temp_dest_coeff1, 1, wide_product_uint64_count, temp_dest_coeff1);

                multiply_uint_uint(temp_dest_coeff1, wide_product_uint64_count, parms_.plain_modulus().pointer(), plain_uint64_count, intermediate_uint64_count, intermediate.get());
                add_uint_uint(intermediate.get(), intermediate_uint64_count, coeff_modulus_div_two_.get(), coeff_uint64_count, false, intermediate_uint64_count, intermediate.get());
                divide_uint_uint_inplace(intermediate.get(), mod_, intermediate_uint64_count, quotient.get(), pool_);
                modulo_uint_inplace(quotient.get(), intermediate_uint64_count, mod_, pool_);

                if (coeff_is_negative)
                {
                    negate_uint_mod(quotient.get(), parms_.coeff_modulus().pointer(), coeff_uint64_count, dest_coeff1);
                }
                else
                {
                    set_uint_uint(quotient.get(), coeff_uint64_count, dest_coeff1);
                }

                temp_dest_coeff1 += wide_product_uint64_count;
                dest_coeff1 += coeff_uint64_count;

                // Dest[2]
                coeff_is_negative = is_high_bit_set_uint(temp_dest_coeff2, wide_product_uint64_count);
                if (coeff_is_negative)
                {
                    negate_uint(temp_dest_coeff2, wide_product_uint64_count, temp_dest_coeff2);
                }

                multiply_uint_uint(temp_dest_coeff2, wide_product_uint64_count, parms_.plain_modulus().pointer(), plain_uint64_count, intermediate_uint64_count, intermediate.get());
                add_uint_uint(intermediate.get(), intermediate_uint64_count, coeff_modulus_div_two_.get(), coeff_uint64_count, false, intermediate_uint64_count, intermediate.get());
                divide_uint_uint_inplace(intermediate.get(), mod_, intermediate_uint64_count, quotient.get(), pool_);
                modulo_uint_inplace(quotient.get(), intermediate_uint64_count, mod_, pool_);

                if (coeff_is_negative)
                {
                    negate_uint_mod(quotient.get(), parms_.coeff_modulus().pointer(), coeff_uint64_count, dest_coeff2);
                }
                else
                {
                    set_uint_uint(quotient.get(), coeff_uint64_count, dest_coeff2);
                }

                temp_dest_coeff2 += wide_product_uint64_count;
                dest_coeff2 += coeff_uint64_count;
            }

            return;
        }
        else
        {
            // This branch should never be reached
            throw logic_error("invalid encryption parameters");
        }
    }

    void Evaluator::relinearize(const Ciphertext &encrypted, Ciphertext &destination, int destination_size)
    {
        const BigPolyArray &encrypted_array = encrypted.get_array();
        BigPolyArray &destination_array = destination.get_mutable_array();

        // Extract encryption parameters.
        int coeff_count = parms_.poly_modulus().coeff_count();
        int coeff_bit_count = parms_.coeff_modulus().bit_count();
        int coeff_uint64_count = parms_.coeff_modulus().uint64_count();
        int encrypted_count = encrypted_array.size();

        // Verify parameters.
        if (encrypted.hash_block_ != parms_.get_hash_block())
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (destination_size < 2 || destination_size > encrypted_count)
        {
            throw invalid_argument("destination_size must be greater than or equal to 2 and less than or equal to current count");
        }
        if (evaluation_keys_.size() < encrypted_count - 2)
        {
            throw invalid_argument("not enough evaluation keys");
        }

        // If encrypted is already at the desired level, return
        if (destination_size == encrypted_count)
        {
            destination = encrypted;
            return;
        }

        ConstPointer encrypted_safe = duplicate_bigpolyarray_if_needed(encrypted_array, &encrypted == &destination, pool_);

        // Prepare destination
        destination_array.resize(destination_size, coeff_count, coeff_bit_count);
        destination.hash_block_ = parms_.get_hash_block();

        // Relinearize one step at a time
        Pointer temp1(allocate_poly(encrypted_count * coeff_count, coeff_uint64_count, pool_));
        Pointer temp2(allocate_poly((encrypted_count - 1) * coeff_count, coeff_uint64_count, pool_));
        set_uint_uint(encrypted_safe.get(), encrypted_count * coeff_count * coeff_uint64_count, temp1.get());

        // Calculate number of relinearize_one_step calls needed
        int relins_needed = encrypted_count - destination_size;

        // Update temp to store the current result after relinearization
        for (int i = 0; i < relins_needed; i++)
        {
            relinearize_one_step(temp1.get(), encrypted_count, temp2.get(), pool_);
            set_uint_uint(temp2.get(), (encrypted_count - 1) * coeff_count * coeff_uint64_count, temp1.get());
            encrypted_count--;
        }

        // Put the output of final relinearization into destination_array.
        // We assume here that destination has been resized by the calling function.
        set_poly_poly(temp2.get(), destination_size * coeff_count, coeff_uint64_count, destination_array.pointer(0));
    }

    void Evaluator::relinearize_one_step(const uint64_t *encrypted, int encrypted_size, uint64_t *destination, MemoryPool &pool_)
    {
        // Extract encryption parameters.
        int coeff_count = parms_.poly_modulus().coeff_count();
        int coeff_uint64_count = parms_.coeff_modulus().uint64_count();
        int array_poly_uint64_count = coeff_count * coeff_uint64_count;

        // encrypted_array[2], encrypted_array[3], ... encrypted_array[count - 2] all stay the same
        if(encrypted_size > 3)
        {
            set_poly_poly(encrypted + 2 * array_poly_uint64_count, (encrypted_size - 3) * coeff_count, coeff_uint64_count, destination + 2 * array_poly_uint64_count);
        }

        // decompose encrypted_array[count-1] into base w
        // want to create an array of polys, each of whose components i is (encrypted_array[count-1])^(i) - in the notation of FV paper
        Pointer decomp_encrypted_last(allocate_poly(evaluation_keys_.get_data()[0].first.size() * coeff_count, coeff_uint64_count, pool_));

        // Populate one poly at a time.
        // Create a polynomial to store the current decomposition value which will be copied into the array to populate it at the current index.
        int shift = 0;
        for (int j = 0; j < evaluation_keys_.get_data()[0].first.size(); j++)
        {
            const uint64_t *encrypted_coeff = encrypted + (encrypted_size - 1) * array_poly_uint64_count;
            uint64_t *decomp_coeff = decomp_encrypted_last.get() + j * array_poly_uint64_count;

            // Isolate decomposition_bit_count_ bits for each coefficient.
            for (int coeff_index = 0; coeff_index < coeff_count; coeff_index++)
            {
                right_shift_uint(encrypted_coeff, shift, coeff_uint64_count, decomp_coeff);
                filter_highbits_uint(decomp_coeff, coeff_uint64_count, parms_.decomposition_bit_count());

                encrypted_coeff += coeff_uint64_count;
                decomp_coeff += coeff_uint64_count;
            }

            // Increase shift by decomposition_bit_count_ for next iteration.
            shift += parms_.decomposition_bit_count();
        }

        Pointer innerproduct1(allocate_poly(coeff_count, coeff_uint64_count, pool_));
        Pointer innerproduct2(allocate_poly(coeff_count, coeff_uint64_count, pool_));
        set_zero_uint(coeff_uint64_count, get_poly_coeff(innerproduct1.get(), coeff_count - 1, coeff_uint64_count));
        set_zero_uint(coeff_uint64_count, get_poly_coeff(innerproduct2.get(), coeff_count - 1, coeff_uint64_count));

        // destination_array[0] = encrypted_array[0] + < appropriate evk , decomposition of encrypted_array[count-1]>

        if (qualifiers_.enable_ntt)
        {
            ntt_double_dot_product_bigpolyarray_nttbigpolyarrays(decomp_encrypted_last.get(), evaluation_keys_.get_data()[encrypted_size - 3].first.pointer(0), evaluation_keys_.get_data()[encrypted_size - 3].second.pointer(0),
                evaluation_keys_.get_data()[0].first.size(), array_poly_uint64_count, ntt_tables_, innerproduct1.get(), innerproduct2.get(), pool_);
            add_poly_poly_coeffmod(encrypted, innerproduct1.get(), coeff_count, mod_.get(), coeff_uint64_count, destination);
            add_poly_poly_coeffmod(encrypted + array_poly_uint64_count, innerproduct2.get(), coeff_count, mod_.get(), coeff_uint64_count, destination + array_poly_uint64_count);
        }
        else if (!qualifiers_.enable_ntt && qualifiers_.enable_fft)
        {
            nussbaumer_dot_product_bigpolyarray_coeffmod(evaluation_keys_.get_data()[encrypted_size - 3].first.pointer(0), decomp_encrypted_last.get(),
                evaluation_keys_.get_data()[0].first.size(), polymod_, mod_, innerproduct1.get(), pool_);
            add_poly_poly_coeffmod(encrypted, innerproduct1.get(), coeff_count, mod_.get(), coeff_uint64_count, destination);

            nussbaumer_dot_product_bigpolyarray_coeffmod(evaluation_keys_.get_data()[encrypted_size - 3].second.pointer(0), decomp_encrypted_last.get(),
                evaluation_keys_.get_data()[0].first.size(), polymod_, mod_, innerproduct2.get(), pool_);
            add_poly_poly_coeffmod(encrypted + array_poly_uint64_count, innerproduct2.get(), coeff_count, mod_.get(), coeff_uint64_count, destination + array_poly_uint64_count);
        }
        else
        {
            // This branch should never be reached
            throw logic_error("invalid encryption parameters");
        }
    }

    void Evaluator::add_plain(const Ciphertext &encrypted, const Plaintext &plain, Ciphertext &destination)
    {
        const BigPolyArray &encrypted_array = encrypted.get_array();
        const BigPoly &plain_poly = plain.get_poly();
        BigPolyArray &destination_array = destination.get_mutable_array();

        // Extract encryption parameters.
        int coeff_count = parms_.poly_modulus().coeff_count();
        int coeff_bit_count = parms_.coeff_modulus().bit_count();
        int coeff_uint64_count = parms_.coeff_modulus().uint64_count();
        int encrypted_count = encrypted_array.size();
        int plain_coeff_uint64_count = divide_round_up(plain_poly.coeff_bit_count(), bits_per_uint64);

        // Verify parameters.
        if (encrypted.hash_block_ != parms_.get_hash_block())
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
#ifdef _DEBUG
        if (plain_poly.significant_coeff_count() >= coeff_count || !are_poly_coefficients_less_than(plain_poly, BigUInt(parms_.plain_modulus().bit_count(), *parms_.plain_modulus().pointer())))
        {
            throw invalid_argument("plain is too large to be represented by encryption parameters");
        }
#endif
        ConstPointer encrypted_safe = duplicate_bigpolyarray_if_needed(encrypted_array, &encrypted == &destination, pool_);

        // Prepare destination
        destination_array.resize(encrypted_count, coeff_count, coeff_bit_count);
        destination.hash_block_ = parms_.get_hash_block();

        // Multiply plain by scalar coeff_div_plaintext and reposition if in upper-half, store in destination_array[0]
        preencrypt(plain_poly.pointer(), plain_poly.coeff_count(), plain_coeff_uint64_count, destination_array.pointer(0));

        // Add preencrypted-version of plain with encrypted_array[0], store in destination_array[0].
        add_poly_poly_coeffmod(encrypted_safe.get(), destination_array.pointer(0), coeff_count, parms_.coeff_modulus().pointer(), coeff_uint64_count, destination_array.pointer(0));

        // Set the remainder of destination to be as in encrypted_array.
        set_poly_poly(encrypted_safe.get() + coeff_count * coeff_uint64_count, coeff_count * (encrypted_count - 1), coeff_uint64_count, destination_array.pointer(1));
    }

    void Evaluator::sub_plain(const Ciphertext &encrypted, const Plaintext &plain, Ciphertext &destination)
    {
        const BigPolyArray &encrypted_array = encrypted.get_array();
        const BigPoly &plain_poly = plain.get_poly();
        BigPolyArray &destination_array = destination.get_mutable_array();

        // Extract encryption parameters.
        int coeff_count = parms_.poly_modulus().coeff_count();
        int coeff_bit_count = parms_.coeff_modulus().bit_count();
        int coeff_uint64_count = parms_.coeff_modulus().uint64_count();
        int encrypted_count = encrypted_array.size();
        int plain_coeff_uint64_count = divide_round_up(plain_poly.coeff_bit_count(), bits_per_uint64);

        // Verify parameters.
        if (encrypted.hash_block_ != parms_.get_hash_block())
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
#ifdef _DEBUG
        if (plain_poly.significant_coeff_count() >= coeff_count || !are_poly_coefficients_less_than(plain_poly, BigUInt(parms_.plain_modulus().bit_count(), *parms_.plain_modulus().pointer())))
        {
            throw invalid_argument("plain is too large to be represented by encryption parameters");
        }
#endif
        ConstPointer encrypted_safe = duplicate_bigpolyarray_if_needed(encrypted_array, &encrypted == &destination, pool_);
        
        // Prepare destination
        destination_array.resize(encrypted_count, coeff_count, coeff_bit_count);
        destination.hash_block_ = parms_.get_hash_block();

        // Multiply plain by scalar coeff_div_plaintext and reposition if in upper-half, store in destination_array[0]
        preencrypt(plain_poly.pointer(), plain_poly.coeff_count(), plain_coeff_uint64_count, destination_array.pointer(0));

        // Subtract preencrypted-version of plain2 with encrypted_array[0], store in destination_array[0].
        sub_poly_poly_coeffmod(encrypted_safe.get(), destination_array.pointer(0), coeff_count, parms_.coeff_modulus().pointer(), coeff_uint64_count, destination_array.pointer(0));

        // Set the remainder of destination to be as in encrypted_array.
        set_poly_poly(encrypted_safe.get() + coeff_count * coeff_uint64_count, coeff_count * (encrypted_count - 1), coeff_uint64_count, destination_array.pointer(1));
    }

    void Evaluator::multiply_plain(const Ciphertext &encrypted, const Plaintext &plain, Ciphertext &destination)
    {
        const BigPolyArray &encrypted_array = encrypted.get_array();
        const BigPoly &plain_poly = plain.get_poly();
        BigPolyArray &destination_array = destination.get_mutable_array();

        // Extract encryption parameters.
        int coeff_count = parms_.poly_modulus().coeff_count();
        int coeff_bit_count = parms_.coeff_modulus().bit_count();
        int coeff_uint64_count = parms_.coeff_modulus().uint64_count();
        int encrypted_count = encrypted_array.size();

        // Verify parameters.
        if (encrypted.hash_block_ != parms_.get_hash_block())
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (plain_poly.is_zero())
        {
            throw invalid_argument("plain cannot be zero");
        }
#ifdef _DEBUG
        if (plain_poly.significant_coeff_count() >= coeff_count || !are_poly_coefficients_less_than(plain_poly, BigUInt(parms_.plain_modulus().bit_count(), *parms_.plain_modulus().pointer())))
        {
            throw invalid_argument("plain is too large to be represented by encryption parameters");
        }
#endif
        // Prepare destination
        destination_array.resize(encrypted_count, coeff_count, coeff_bit_count);
        destination.hash_block_ = parms_.get_hash_block();

        // Multiplying just by a constant?
        int plain_coeff_count = min(plain_poly.coeff_count(), coeff_count - 1);
        int plain_coeff_uint64_count = plain_poly.coeff_uint64_count();
        const uint64_t *plain_coeff = plain_poly.pointer();
        if (plain_coeff_count == 1)
        {
            Pointer moved2ptr(allocate_uint(coeff_uint64_count, pool_));
            uint64_t *moved2_coeff = moved2ptr.get();
            bool is_upper_half = is_greater_than_or_equal_uint_uint(plain_coeff, plain_coeff_uint64_count, plain_upper_half_threshold_.get(), coeff_uint64_count);
            if (is_upper_half)
            {
                add_uint_uint(plain_coeff, plain_coeff_uint64_count, plain_upper_half_increment_.get(), coeff_uint64_count, 0, coeff_uint64_count, moved2_coeff);
            }
            else
            {
                set_uint_uint(plain_coeff, plain_coeff_uint64_count, coeff_uint64_count, moved2_coeff);
            }
            for (int i = 0; i < encrypted_count; i++)
            {
                multiply_poly_scalar_coeffmod(encrypted_array.pointer(i), coeff_count, moved2_coeff, mod_, destination_array.pointer(i), pool_);
            }

            return;
        }

        // Reposition coefficients.
        Pointer moved2ptr(allocate_poly(coeff_count, coeff_uint64_count, pool_));
        uint64_t *moved2_coeff = moved2ptr.get();
        for (int i = 0; i < plain_coeff_count; i++)
        {
            bool is_upper_half = is_greater_than_or_equal_uint_uint(plain_coeff, plain_coeff_uint64_count, plain_upper_half_threshold_.get(), coeff_uint64_count);
            if (is_upper_half)
            {
                add_uint_uint(plain_coeff, plain_coeff_uint64_count, plain_upper_half_increment_.get(), coeff_uint64_count, 0, coeff_uint64_count, moved2_coeff);
            }
            else
            {
                set_uint_uint(plain_coeff, plain_coeff_uint64_count, coeff_uint64_count, moved2_coeff);
            }
            moved2_coeff += coeff_uint64_count;
            plain_coeff += plain_coeff_uint64_count;
        }
        set_zero_poly(coeff_count - plain_coeff_count, coeff_uint64_count, moved2_coeff);

        // Need to multiply each component in encrypted with moved2ptr (plain poly)
        if (qualifiers_.enable_ntt)
        {
            // Transform plain poly only once
            Pointer copy_operand1(allocate_poly(ntt_tables_.coeff_count(), ntt_tables_.coeff_uint64_count(), pool_));
            ntt_negacyclic_harvey(moved2ptr.get(), ntt_tables_, pool_);
            for (int i = 0; i < encrypted_count; i++)
            {
                ntt_multiply_poly_nttpoly(encrypted_array.pointer(i), moved2ptr.get(), ntt_tables_, destination_array.pointer(i), pool_);
            }
        }
        else if (!qualifiers_.enable_ntt && qualifiers_.enable_fft)
        {
            // Improve this path
            for (int i = 0; i < encrypted_count; i++)
            {
                nussbaumer_multiply_poly_poly_coeffmod(encrypted_array.pointer(i), moved2ptr.get(), polymod_.coeff_count_power_of_two(), mod_, destination_array.pointer(i), pool_);
            }
        }
        else
        {
            // This branch should never be reached
            throw logic_error("invalid encryption parameters");
        }
    }

    void Evaluator::preencrypt(const uint64_t *plain, int plain_coeff_count, int plain_coeff_uint64_count, uint64_t *destination)
    {
        // Extract encryption parameters.
        int coeff_count = parms_.poly_modulus().coeff_count();
        int coeff_uint64_count = parms_.coeff_modulus().uint64_count();

        if (plain_coeff_count > coeff_count)
        {
            plain_coeff_count = coeff_count;
        }

        // Multiply plain by scalar coeff_div_plaintext and reposition if in upper-half.
        if (plain == destination)
        {
            // If plain and destination are same poly, then need another storage for multiply output.
            Pointer temp(allocate_uint(coeff_uint64_count, pool_));
            for (int i = 0; i < plain_coeff_count; i++)
            {
                multiply_uint_uint(plain, plain_coeff_uint64_count, coeff_div_plain_modulus_.get(), coeff_uint64_count, coeff_uint64_count, temp.get());
                bool is_upper_half = is_greater_than_or_equal_uint_uint(temp.get(), upper_half_threshold_.get(), coeff_uint64_count);
                if (is_upper_half)
                {
                    add_uint_uint(temp.get(), upper_half_increment_.get(), coeff_uint64_count, destination);
                }
                else
                {
                    set_uint_uint(temp.get(), coeff_uint64_count, destination);
                }
                plain += plain_coeff_uint64_count;
                destination += coeff_uint64_count;
            }
        }
        else
        {
            for (int i = 0; i < plain_coeff_count; i++)
            {
                //Multiply plain by coeff_div_plain_modulus and put the result in destination
                multiply_uint_uint(plain, plain_coeff_uint64_count, coeff_div_plain_modulus_.get(), coeff_uint64_count, coeff_uint64_count, destination);

                // check if destination >= upper half threshold
                bool is_upper_half = is_greater_than_or_equal_uint_uint(destination, upper_half_threshold_.get(), coeff_uint64_count);
                if (is_upper_half)
                {
                    add_uint_uint(destination, upper_half_increment_.get(), coeff_uint64_count, destination);
                }
                plain += plain_coeff_uint64_count;
                destination += coeff_uint64_count;
            }
        }

        // Zero any remaining coefficients.
        for (int i = plain_coeff_count; i < coeff_count; i++)
        {
            set_zero_uint(coeff_uint64_count, destination);
            destination += coeff_uint64_count;
        }
    }

    Ciphertext Evaluator::multiply_many(vector<Ciphertext> encrypteds)
    {
        size_t original_size = encrypteds.size();

        // Verify parameters.
        if (original_size == 0)
        {
            throw invalid_argument("encrypteds vector must not be empty");
        }

        // If there is only one ciphertext, return it after checking validity.
        if (original_size == 1)
        {
            // Verify parameters.
            if (encrypteds[0].hash_block_ != parms_.get_hash_block())
            {
                throw invalid_argument("encrypteds is not valid for encryption parameters");
            }
            return encrypteds[0];
        }

        // Repeatedly multiply and add to the back of the vector until the end is reached
        for (size_t i = 0; i < encrypteds.size() - 1; i += 2)
        {
            encrypteds.emplace_back(relinearize(multiply(encrypteds[i], encrypteds[i + 1])));
        }

        return encrypteds[encrypteds.size() - 1];
    }

    void Evaluator::exponentiate(const Ciphertext &encrypted, uint64_t exponent, Ciphertext &destination)
    {
        // Verify parameters.
        if (encrypted.hash_block_ != parms_.get_hash_block())
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (exponent == 0)
        {
            throw invalid_argument("exponent cannot be 0");
        }

        if (exponent == 1)
        {
            destination = encrypted;
            return;
        }

        vector<Ciphertext> exp_vector(exponent, encrypted);
        destination = multiply_many(exp_vector);
    }

    void Evaluator::transform_to_ntt(Plaintext &plain)
    {
        BigPoly &plain_poly = plain.get_poly();

        // Do the encryption parameters support NTT?
        if (!qualifiers_.enable_ntt)
        {
            throw logic_error("encryption parameters do not support NTT");
        }

        // Extract encryption parameters.
        int coeff_count = parms_.poly_modulus().coeff_count();
        int coeff_bit_count = parms_.coeff_modulus().bit_count();
        int coeff_uint64_count = parms_.coeff_modulus().uint64_count();
        int plain_coeff_count = plain_poly.coeff_count();

        // Verify parameters.
        if (plain_poly.coeff_count() > coeff_count || plain_poly.coeff_bit_count() > coeff_bit_count)
        {
            throw invalid_argument("plain is not valid for encryption parameters");
        }
#ifdef _DEBUG
        if (plain_poly.significant_coeff_count() >= coeff_count || !are_poly_coefficients_less_than(plain_poly, BigUInt(parms_.plain_modulus().bit_count(), *parms_.plain_modulus().pointer())))
        {
            throw invalid_argument("plain is too large to be represented by encryption parameters");
        }
#endif
        plain_poly.resize(coeff_count, coeff_bit_count);

        uint64_t *plain_coeff = plain_poly.pointer();
        for (int i = 0; i < plain_coeff_count; i++)
        {
            bool is_upper_half = is_greater_than_or_equal_uint_uint(plain_coeff, plain_upper_half_threshold_.get(), coeff_uint64_count);
            if (is_upper_half)
            {
                add_uint_uint(plain_coeff, plain_upper_half_increment_.get(), coeff_uint64_count, plain_coeff);
            }
            plain_coeff += coeff_uint64_count;
        }
        set_zero_poly(coeff_count - plain_coeff_count, coeff_uint64_count, plain_coeff);
        
        // Transform to NTT domain
        ntt_negacyclic_harvey(plain_poly.pointer(), ntt_tables_, pool_);
    }

    void Evaluator::transform_to_ntt(Ciphertext &encrypted)
    {
        BigPolyArray &encrypted_array = encrypted.get_mutable_array();

        // Do the encryption parameters support NTT?
        if (!qualifiers_.enable_ntt)
        {
            throw logic_error("encryption parameters do not support NTT");
        }

        // Verify parameters.
        if (encrypted.hash_block_ != parms_.get_hash_block())
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }

        // Transform each polynomial to NTT domain
        for (int i = 0; i < encrypted_array.size(); i++)
        {
            ntt_negacyclic_harvey(encrypted_array.pointer(i), ntt_tables_, pool_);
        }
    }

    void Evaluator::transform_from_ntt(Plaintext &plain_ntt)
    {
        BigPoly &plain_ntt_poly = plain_ntt.get_poly();

        // Do the encryption parameters support NTT?
        if (!qualifiers_.enable_ntt)
        {
            throw logic_error("encryption parameters do not support NTT");
        }

        // Extract encryption parameters.
        int coeff_count = parms_.poly_modulus().coeff_count();
        int coeff_bit_count = parms_.coeff_modulus().bit_count();
        int coeff_uint64_count = parms_.coeff_modulus().uint64_count();

        // Verify parameters.
        if (plain_ntt_poly.coeff_count() != coeff_count || plain_ntt_poly.coeff_bit_count() != coeff_bit_count)
        {
            throw invalid_argument("plain_ntt is not valid for encryption parameters");
        }
#ifdef _DEBUG
        if (plain_ntt_poly.significant_coeff_count() == coeff_count || !are_poly_coefficients_less_than(plain_ntt_poly, parms_.coeff_modulus()))
        {
            throw invalid_argument("plain_ntt is not valid for encryption parameters");
        }
#endif
        // Transform back to polynomial domain
        inverse_ntt_negacyclic_harvey(plain_ntt_poly.pointer(), ntt_tables_, pool_);

        uint64_t *plain_coeff = plain_ntt_poly.pointer();
        for (int i = 0; i < coeff_count; i++)
        {
            bool is_upper_half = is_greater_than_or_equal_uint_uint(plain_coeff, upper_half_threshold_.get(), coeff_uint64_count);
            if (is_upper_half)
            {
                sub_uint_uint(plain_coeff, plain_upper_half_increment_.get(), coeff_uint64_count, plain_coeff);
            }
            plain_coeff += coeff_uint64_count;
        }

        // Verify output.
#ifdef _DEBUG
        if (plain_ntt_poly.significant_coeff_count() >= coeff_count || !are_poly_coefficients_less_than(plain_ntt_poly, BigUInt(parms_.plain_modulus().bit_count(), *parms_.plain_modulus().pointer())))
        {
            throw invalid_argument("plain_ntt is not valid for encryption parameters");
        }
#endif
    }

    void Evaluator::transform_from_ntt(Ciphertext &encrypted_ntt)
    {
        BigPolyArray &encrypted_ntt_array = encrypted_ntt.get_mutable_array();

        // Do the encryption parameters support NTT?
        if (!qualifiers_.enable_ntt)
        {
            throw logic_error("encryption parameters do not support NTT");
        }

        // Verify parameters.
        if (encrypted_ntt.hash_block_ != parms_.get_hash_block())
        {
            throw invalid_argument("encrypted_ntt is not valid for encryption parameters");
        }

        // Transform each polynomial from NTT domain
        for (int i = 0; i < encrypted_ntt.size(); i++)
        {
            inverse_ntt_negacyclic_harvey(encrypted_ntt_array.pointer(i), ntt_tables_, pool_);
        }
    }

    void Evaluator::multiply_plain_ntt(const Ciphertext &encrypted_ntt, const Plaintext &plain_ntt, Ciphertext &destination_ntt)
    {
        const BigPolyArray &encrypted_ntt_array = encrypted_ntt.get_array();
        const BigPoly &plain_ntt_poly = plain_ntt.get_poly();
        BigPolyArray &destination_ntt_array = destination_ntt.get_mutable_array();

        // Do the encryption parameters support NTT?
        if (!qualifiers_.enable_ntt)
        {
            throw logic_error("encryption parameters do not support NTT");
        }

        // Extract encryption parameters.
        int coeff_count = parms_.poly_modulus().coeff_count();
        int coeff_bit_count = parms_.coeff_modulus().bit_count();
        int coeff_uint64_count = parms_.coeff_modulus().uint64_count();
        int encrypted_count = encrypted_ntt.size();

        // Verify parameters.
        if (encrypted_ntt.hash_block_ != parms_.get_hash_block())
        {
            throw invalid_argument("encrypted_ntt is not valid for encryption parameters");
        }
        if (plain_ntt_poly.is_zero())
        {
            throw invalid_argument("plain cannot be zero");
        }
#ifdef _DEBUG
        if (plain_ntt_poly.significant_coeff_count() >= coeff_count || !are_poly_coefficients_less_than(plain_ntt_poly, parms_.coeff_modulus()))
        {
            throw invalid_argument("plain_ntt is too large to be represented by encryption parameters");
        }
#endif
        // Prepare destination
        destination_ntt_array.resize(encrypted_count, coeff_count, coeff_bit_count);
        destination_ntt.hash_block_ = parms_.get_hash_block();

        Pointer temp(allocate_poly(coeff_count, coeff_uint64_count, pool_));
        for (int i = 0; i < encrypted_count; i++)
        {
            set_poly_poly(encrypted_ntt_array.pointer(i), coeff_count, coeff_uint64_count, temp.get());
            dyadic_product_coeffmod(temp.get(), plain_ntt_poly.pointer(), coeff_count - 1, ntt_tables_.modulus(), destination_ntt_array.pointer(i), pool_);
        }
    }

    Evaluator::Evaluator(const SEALContext &context, const EvaluationKeys &evaluation_keys, const MemoryPoolHandle &pool) :
        pool_(pool), parms_(context.get_parms()), qualifiers_(context.get_qualifiers()),
        ntt_tables_(pool_), evaluation_keys_(evaluation_keys)
    {
        // Verify parameters
        if (!qualifiers_.parameters_set)
        {
            throw invalid_argument("encryption parameters are not set correctly");
        }
        if (evaluation_keys.size() > 0)
        {
            if (evaluation_keys.get_hash_block() != parms_.get_hash_block())
            {
                throw invalid_argument("evaluation keys are not valid for encryption parameters");
            }
        }

        int coeff_count = parms_.poly_modulus().coeff_count();
        int poly_coeff_uint64_count = parms_.poly_modulus().coeff_uint64_count();
        int coeff_bit_count = parms_.coeff_modulus().bit_count();
        int coeff_uint64_count = parms_.coeff_modulus().uint64_count();
        // Calculate coeff_modulus / plain_modulus.
        coeff_div_plain_modulus_ = allocate_uint(coeff_uint64_count, pool_);
        ConstPointer wide_plain_modulus(duplicate_uint_if_needed(parms_.plain_modulus().pointer(), parms_.plain_modulus().uint64_count(), coeff_uint64_count, false, pool_));
        Pointer temp(allocate_uint(coeff_uint64_count, pool_));
        divide_uint_uint(parms_.coeff_modulus().pointer(), wide_plain_modulus.get(), coeff_uint64_count, coeff_div_plain_modulus_.get(), temp.get(), pool_);

        // Calculate (plain_modulus + 1) / 2.
        plain_upper_half_threshold_ = allocate_uint(coeff_uint64_count, pool_);
        half_round_up_uint(wide_plain_modulus.get(), coeff_uint64_count, plain_upper_half_threshold_.get());

        // Calculate coeff_modulus - plain_modulus.
        plain_upper_half_increment_ = allocate_uint(coeff_uint64_count, pool_);
        sub_uint_uint(parms_.coeff_modulus().pointer(), wide_plain_modulus.get(), coeff_uint64_count, plain_upper_half_increment_.get());

        // Calculate (plain_modulus + 1) / 2 * coeff_div_plain_modulus.
        upper_half_threshold_ = allocate_uint(coeff_uint64_count, pool_);
        multiply_truncate_uint_uint(plain_upper_half_threshold_.get(), coeff_div_plain_modulus_.get(), coeff_uint64_count, upper_half_threshold_.get());

        // Calculate upper_half_increment.
        upper_half_increment_ = allocate_uint(coeff_uint64_count, pool_);
        multiply_truncate_uint_uint(wide_plain_modulus.get(), coeff_div_plain_modulus_.get(), coeff_uint64_count, upper_half_increment_.get());
        sub_uint_uint(parms_.coeff_modulus().pointer(), upper_half_increment_.get(), coeff_uint64_count, upper_half_increment_.get());

        // Calculate coeff_modulus_ / 2.
        coeff_modulus_div_two_ = allocate_uint(coeff_uint64_count, pool_);
        right_shift_uint(parms_.coeff_modulus().pointer(), 1, coeff_uint64_count, coeff_modulus_div_two_.get());

        // Wide enough to store product of two polynomials and reserve one bit for sign (as result of FFT multiplication)
        product_coeff_bit_count_ = 2 * coeff_bit_count + get_significant_bit_count(static_cast<uint64_t>(coeff_count)) + 1;

        // Initialize moduli.
        polymod_ = PolyModulus(parms_.poly_modulus().pointer(), coeff_count, poly_coeff_uint64_count);
        mod_ = Modulus(parms_.coeff_modulus().pointer(), coeff_uint64_count, pool_);

        if (qualifiers_.enable_ntt)
        {
            // Copy over NTT tables (switching to local pool)
            ntt_tables_ = context.ntt_tables_;
        }
    }
}

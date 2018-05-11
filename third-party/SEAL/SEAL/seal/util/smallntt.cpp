#include "seal/util/smallntt.h"
#include "seal/util/polyarith.h"
#include "seal/util/uintarith.h"
#include "seal/util/modulus.h"
#include "seal/smallmodulus.h"
#include "seal/util/uintarithsmallmod.h"
#include "seal/util/defines.h"
#include <algorithm>

using namespace std;

namespace seal
{
    namespace util
    {
        SmallNTTTables::SmallNTTTables(const MemoryPoolHandle &pool) : pool_(pool)
        {
        }

        SmallNTTTables::SmallNTTTables(int coeff_count_power, const SmallModulus &modulus, const MemoryPoolHandle &pool) :
            pool_(pool)
        {
            generate(coeff_count_power, modulus);
        }

        void SmallNTTTables::reset()
        {
            generated_ = false;
            modulus_ = SmallModulus();
            root_ = 0;
            root_powers_.release();
            scaled_root_powers_.release();
            inv_root_powers_.release();
            scaled_inv_root_powers_.release();
            inv_root_powers_div_two_.release();
            scaled_inv_root_powers_div_two_.release();
            inv_degree_modulo_ = 0;
            coeff_count_power_ = 0;
            coeff_count_ = 0;
        }

        bool SmallNTTTables::generate(int coeff_count_power, const SmallModulus &modulus)
        {
            reset();

            coeff_count_power_ = coeff_count_power;
            coeff_count_ = 1 << coeff_count_power_;

            // Allocate memory for the tables
            root_powers_ = allocate_uint(coeff_count_, pool_);
            inv_root_powers_ = allocate_uint(coeff_count_, pool_);
            scaled_root_powers_ = allocate_uint(coeff_count_, pool_);
            scaled_inv_root_powers_ = allocate_uint(coeff_count_, pool_);
            inv_root_powers_div_two_ = allocate_uint(coeff_count_, pool_);
            scaled_inv_root_powers_div_two_ = allocate_uint(coeff_count_, pool_);
            modulus_ = modulus;

            // We defer parameter checking to try_minimal_primitive_root(...)
            if (!try_minimal_primitive_root(2 * coeff_count_, modulus_, root_))
            {
                reset();
                return false;
            }

            uint64_t inverse_root;
            if (!try_invert_uint_mod(root_, modulus_, inverse_root))
            {
                reset();
                return false;
            }

            // Populate the tables storing (scaled version of) powers of root mod q in bit-scrambled order.  
            ntt_powers_of_primitive_root(root_, root_powers_.get());
            ntt_scale_powers_of_primitive_root(root_powers_.get(), scaled_root_powers_.get());

            // Populate the tables storing (scaled version of) powers of (root)^{-1} mod q in bit-scrambled order.  
            ntt_powers_of_primitive_root(inverse_root, inv_root_powers_.get());
            ntt_scale_powers_of_primitive_root(inv_root_powers_.get(), scaled_inv_root_powers_.get());

            // Populate the tables storing (scaled version of ) 2 times powers of roots^-1 mod q  in bit-scrambled order. 
            for (int i = 0; i < coeff_count_; i++)
            {
                inv_root_powers_div_two_[i] = div2_uint_mod(inv_root_powers_[i], modulus_);
            }
            ntt_scale_powers_of_primitive_root(inv_root_powers_div_two_.get(), scaled_inv_root_powers_div_two_.get());

            // Last compute n^(-1) modulo q. 
            uint64_t degree_uint = static_cast<uint64_t>(coeff_count_);
            generated_ = try_invert_uint_mod(degree_uint, modulus_, inv_degree_modulo_);

            if (!generated_)
            {
                reset();
                return false;
            }
            return true;
        }

        SmallNTTTables::SmallNTTTables(const SmallNTTTables &copy) : pool_(copy.pool_), generated_(copy.generated_), coeff_count_power_(copy.coeff_count_power_),
            coeff_count_(copy.coeff_count_), modulus_(copy.modulus_), root_(copy.root_), inv_degree_modulo_(copy.inv_degree_modulo_)
        {
            if (generated_)
            {
                // Allocate space and copy tables
                root_powers_ = allocate_uint(coeff_count_, pool_);
                set_uint_uint(copy.root_powers_.get(), coeff_count_, root_powers_.get());

                inv_root_powers_ = allocate_uint(coeff_count_, pool_);
                set_uint_uint(copy.inv_root_powers_.get(), coeff_count_, inv_root_powers_.get());

                scaled_root_powers_ = allocate_uint(coeff_count_, pool_);
                set_uint_uint(copy.scaled_root_powers_.get(), coeff_count_, scaled_root_powers_.get());

                scaled_inv_root_powers_ = allocate_uint(coeff_count_, pool_);
                set_uint_uint(copy.scaled_inv_root_powers_.get(), coeff_count_, scaled_inv_root_powers_.get());

                inv_root_powers_div_two_ = allocate_uint(coeff_count_, pool_);
                set_uint_uint(copy.inv_root_powers_div_two_.get(), coeff_count_, inv_root_powers_div_two_.get());

                scaled_inv_root_powers_div_two_ = allocate_uint(coeff_count_, pool_);
                set_uint_uint(copy.scaled_inv_root_powers_div_two_.get(), coeff_count_, scaled_inv_root_powers_div_two_.get());
            }
        }
        
        SmallNTTTables &SmallNTTTables::operator =(const SmallNTTTables &assign)
        {
            // Check for self-assignment
            if (this == &assign)
            {
                return *this;
            }

            generated_ = assign.generated_;
            coeff_count_power_ = assign.coeff_count_power_;
            coeff_count_ = assign.coeff_count_;

            if (generated_)
            {
                // Copy simple values
                modulus_ = assign.modulus_;
                root_ = assign.root_;
                inv_degree_modulo_ = assign.inv_degree_modulo_;

                // Allocate space and copy tables
                root_powers_ = allocate_uint(coeff_count_, pool_);
                set_uint_uint(assign.root_powers_.get(), coeff_count_, root_powers_.get());

                inv_root_powers_ = allocate_uint(coeff_count_, pool_);
                set_uint_uint(assign.inv_root_powers_.get(), coeff_count_, inv_root_powers_.get());

                scaled_root_powers_ = allocate_uint(coeff_count_, pool_);
                set_uint_uint(assign.scaled_root_powers_.get(), coeff_count_, scaled_root_powers_.get());

                scaled_inv_root_powers_ = allocate_uint(coeff_count_, pool_);
                set_uint_uint(assign.scaled_inv_root_powers_.get(), coeff_count_, scaled_inv_root_powers_.get());

                inv_root_powers_div_two_ = allocate_uint(coeff_count_, pool_);
                set_uint_uint(assign.inv_root_powers_div_two_.get(), coeff_count_, inv_root_powers_div_two_.get());

                scaled_inv_root_powers_div_two_ = allocate_uint(coeff_count_, pool_);
                set_uint_uint(assign.scaled_inv_root_powers_div_two_.get(), coeff_count_, scaled_inv_root_powers_div_two_.get());
            }

            return *this;
        }

        void SmallNTTTables::ntt_powers_of_primitive_root(uint64_t root, uint64_t *destination) const
        {
            uint64_t *destination_start = destination;
            *destination_start = 1;
            for (int i = 1; i < coeff_count_; i++)
            {
                uint64_t *next_destination = destination_start + (reverse_bits(static_cast<uint32_t>(i), coeff_count_power_));
                *next_destination = multiply_uint_uint_mod(*destination, root, modulus_);
                destination = next_destination;
            }
        }

        // compute floor ( input * beta /q ), where beta is a 64k power of 2 and  0 < q < beta. 
        void SmallNTTTables::ntt_scale_powers_of_primitive_root(const uint64_t *input, uint64_t *destination) const
        {
            for (int i = 0; i < coeff_count_; i++, input++, destination++)
            {
                uint64_t wide_quotient[2]{ 0 };
                uint64_t wide_coeff[2]{ 0, *input };
                divide_uint128_uint64_inplace(wide_coeff, modulus_.value(), wide_quotient);
                *destination = wide_quotient[0];
            }
        }

        /**
        This function computes in-place the negacyclic NTT. The input is a polynomial a of degree n in R_q,
        where n is assumed to be a power of 2 and q is a prime such that q = 1 (mod 2n).

        The output is a vector A such that the following hold:
        A[j] =  a(psi**(2*bit_reverse(j) + 1)), 0 <= j < n.

        For details, see Michael Naehrig and Patrick Longa.
        */
        void ntt_negacyclic_harvey_lazy(uint64_t *operand, const SmallNTTTables &tables)
        {
            uint64_t modulus = tables.modulus().value();
            uint64_t two_times_modulus = modulus * 2;
            
            // Return the NTT in scrambled order
            int n = 1 << tables.coeff_count_power();
            int t = n >> 1;
            for (int m = 1; m < n; m <<= 1)
            {
                if (t >= 4)
                {
                    for (int i = 0; i < m; i++)
                    {
                        int j1 = 2 * i * t;
                        int j2 = j1 + t;
                        const uint64_t W = tables.get_from_root_powers(m + i);
                        const uint64_t Wprime = tables.get_from_scaled_root_powers(m + i);

                        uint64_t *X = operand + j1;
                        uint64_t *Y = X + t;
                        uint64_t currX;
                        uint64_t Q;
                        for (int j = j1; j < j2; j += 4)
                        {
                            currX = *X - (two_times_modulus & static_cast<uint64_t>(-static_cast<int64_t>(*X >= two_times_modulus)));
                            multiply_uint64_hw64(Wprime, *Y, &Q);
                            Q = *Y * W - Q * modulus;
                            *X++ = currX + Q;
                            *Y++ = currX + (two_times_modulus - Q);

                            currX = *X - (two_times_modulus & static_cast<uint64_t>(-static_cast<int64_t>(*X >= two_times_modulus)));
                            multiply_uint64_hw64(Wprime, *Y, &Q);
                            Q = *Y * W - Q * modulus;
                            *X++ = currX + Q;
                            *Y++ = currX + (two_times_modulus - Q);

                            currX = *X - (two_times_modulus & static_cast<uint64_t>(-static_cast<int64_t>(*X >= two_times_modulus)));
                            multiply_uint64_hw64(Wprime, *Y, &Q);
                            Q = *Y * W - Q * modulus;
                            *X++ = currX + Q;
                            *Y++ = currX + (two_times_modulus - Q);

                            currX = *X - (two_times_modulus & static_cast<uint64_t>(-static_cast<int64_t>(*X >= two_times_modulus)));
                            multiply_uint64_hw64(Wprime, *Y, &Q);
                            Q = *Y * W - Q * modulus;
                            *X++ = currX + Q;
                            *Y++ = currX + (two_times_modulus - Q);
                        }
                    }
                }
                else
                {
                    for (int i = 0; i < m; i++)
                    {
                        int j1 = 2 * i * t;
                        int j2 = j1 + t;
                        const uint64_t W = tables.get_from_root_powers(m + i);
                        const uint64_t Wprime = tables.get_from_scaled_root_powers(m + i);

                        uint64_t *X = operand + j1;
                        uint64_t *Y = X + t;
                        uint64_t currX;
                        uint64_t Q;
                        for (int j = j1; j < j2; j++)
                        {
                            // The Harvey butterfly: assume X, Y in [0, 2p), and return X', Y' in [0, 2p).
                            // X', Y' = X + WY, X - WY (mod p).
                            currX = *X - (two_times_modulus & static_cast<uint64_t>(-static_cast<int64_t>(*X >= two_times_modulus)));
                            multiply_uint64_hw64(Wprime, *Y, &Q);
                            Q = W * *Y - Q * modulus;
                            *X++ = currX + Q;
                            *Y++ = currX + (two_times_modulus - Q);
                        }
                    }
                }
                t >>= 1;
            }
        }

        // Inverse negacyclic NTT using Harvey's butterfly. (See Patrick Longa and Michael Naehrig). 
        void inverse_ntt_negacyclic_harvey_lazy(uint64_t *operand, const SmallNTTTables &tables)
        {
            uint64_t modulus = tables.modulus().value();
            uint64_t two_times_modulus = modulus * 2;

            // return the bit-reversed order of NTT. 
            int n = 1 << tables.coeff_count_power();
            int t = 1;

            for (int m = n; m > 1; m >>= 1)
            {
                int j1 = 0;
                int h = m >> 1;
                if (t >= 4)
                {
                    for (int i = 0; i < h; i++)
                    {
                        int j2 = j1 + t;
                        // Need the powers of phi^{-1} in bit-reversed order
                        const uint64_t W = tables.get_from_inv_root_powers_div_two(h + i);
                        const uint64_t Wprime = tables.get_from_scaled_inv_root_powers_div_two(h + i);

                        uint64_t *U = operand + j1;
                        uint64_t *V = U + t;
                        uint64_t currU;
                        uint64_t T;
                        uint64_t H;
                        for (int j = j1; j < j2; j += 4)
                        {
                            T = two_times_modulus - *V + *U;
                            currU = *U + *V - (two_times_modulus & static_cast<uint64_t>(-static_cast<int64_t>((*U << 1) >= T)));
                            *U++ = (currU + (modulus & static_cast<uint64_t>(-static_cast<int64_t>(T & 1)))) >> 1;
                            multiply_uint64_hw64(Wprime, T, &H);
                            *V++ = T * W - H * modulus;

                            T = two_times_modulus - *V + *U;
                            currU = *U + *V - (two_times_modulus & static_cast<uint64_t>(-static_cast<int64_t>((*U << 1) >= T)));
                            *U++ = (currU + (modulus & static_cast<uint64_t>(-static_cast<int64_t>(T & 1)))) >> 1;
                            multiply_uint64_hw64(Wprime, T, &H);
                            *V++ = T * W - H * modulus;

                            T = two_times_modulus - *V + *U;
                            currU = *U + *V - (two_times_modulus & static_cast<uint64_t>(-static_cast<int64_t>((*U << 1) >= T)));
                            *U++ = (currU + (modulus & static_cast<uint64_t>(-static_cast<int64_t>(T & 1)))) >> 1;
                            multiply_uint64_hw64(Wprime, T, &H);
                            *V++ = T * W - H * modulus;

                            T = two_times_modulus - *V + *U;
                            currU = *U + *V - (two_times_modulus & static_cast<uint64_t>(-static_cast<int64_t>((*U << 1) >= T)));
                            *U++ = (currU + (modulus & static_cast<uint64_t>(-static_cast<int64_t>(T & 1)))) >> 1;
                            multiply_uint64_hw64(Wprime, T, &H);
                            *V++ = T * W - H * modulus;
                        }
                        j1 += (t << 1);
                    }
                }
                else
                {
                    for (int i = 0; i < h; i++)
                    {
                        int j2 = j1 + t;
                        // Need the powers of  phi^{-1} in bit-reversed order
                        const uint64_t W = tables.get_from_inv_root_powers_div_two(h + i);
                        const uint64_t Wprime = tables.get_from_scaled_inv_root_powers_div_two(h + i);

                        uint64_t *U = operand + j1;
                        uint64_t *V = U + t;
                        uint64_t currU;
                        uint64_t T;
                        uint64_t H;
                        for (int j = j1; j < j2; j++)
                        {
                            currU = *U;
                            // U = x[i], V = x[i+m]

                            // Compute U - V + 2q
                            T = two_times_modulus - *V + *U;

                            // Cleverly check whether currU + currV >= two_times_modulus
                            currU = *U + *V - (two_times_modulus & static_cast<uint64_t>(-static_cast<int64_t>((*U << 1) >= T)));

                            // Need to make it so that div2_uint_mod takes values that are > q. 
                            //div2_uint_mod(U, modulusptr, coeff_uint64_count, U); 
                            // We use also the fact that parity of currU is same as parity of T.
                            // Since our modulus is always so small that currU + masked_modulus < 2^64,
                            // we never need to worry about wrapping around when adding masked_modulus.
                            //uint64_t masked_modulus = modulus & static_cast<uint64_t>(-static_cast<int64_t>(T & 1));
                            //uint64_t carry = add_uint64(currU, masked_modulus, 0, &currU);
                            //currU += modulus & static_cast<uint64_t>(-static_cast<int64_t>(T & 1));
                            *U++ = (currU + (modulus & static_cast<uint64_t>(-static_cast<int64_t>(T & 1)))) >> 1;

                            multiply_uint64_hw64(Wprime, T, &H);
                            // effectively, the next two multiply perform multiply modulo beta = 2**wordsize. 
                            *V++ = W * T - H * modulus;
                        }
                        j1 += (t << 1);
                    }
                }
                t <<= 1;
            }
        }
    }
}

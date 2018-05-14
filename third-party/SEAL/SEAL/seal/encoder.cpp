#include <stdexcept>
#include <algorithm>
#include <cmath>
#include "seal/encoder.h"
#include "seal/util/common.h"
#include "seal/util/uintarith.h"
#include "seal/util/polyarith.h"

using namespace std;
using namespace seal::util;

namespace seal
{
    BinaryEncoder::BinaryEncoder(const SmallModulus &plain_modulus, const MemoryPoolHandle &pool) :
        pool_(pool),
        plain_modulus_(plain_modulus),
        coeff_neg_threshold_((plain_modulus.value() + 1) >> 1),
        neg_one_(plain_modulus_.value() - 1)
    {
        if (plain_modulus.bit_count() <= 1)
        {
            throw invalid_argument("plain_modulus must be at least 2");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }
    }

    Plaintext BinaryEncoder::encode(uint64_t value)
    {
        Plaintext result;
        encode(value, result);
        return result;
    }

    void BinaryEncoder::encode(uint64_t value, Plaintext &destination)
    {
        int encode_coeff_count = get_significant_bit_count(value);
        destination.resize(encode_coeff_count);
        destination.set_zero();

        int coeff_index = 0;
        while (value != 0)
        {
            if ((value & 1) != 0)
            {
                destination[coeff_index] = 1;
            }
            value >>= 1;
            coeff_index++;
        }
    }

    Plaintext BinaryEncoder::encode(int64_t value)
    {
        Plaintext result;
        encode(value, result);
        return result;
    }

    void BinaryEncoder::encode(int64_t value, Plaintext &destination)
    {
        if (value < 0)
        {
            uint64_t pos_value = static_cast<uint64_t>(-value);
            int encode_coeff_count = get_significant_bit_count(pos_value);
            destination.resize(encode_coeff_count);
            destination.set_zero();

            int coeff_index = 0;
            while (pos_value != 0)
            {
                if ((pos_value & 1) != 0)
                {
                    destination[coeff_index] = neg_one_;
                }
                pos_value >>= 1;
                coeff_index++;
            }
        }
        else
        {
            encode(static_cast<uint64_t>(value), destination);
        }
    }

    Plaintext BinaryEncoder::encode(const BigUInt &value)
    {
        Plaintext result;
        encode(value, result);
        return result;
    }

    void BinaryEncoder::encode(const BigUInt &value, Plaintext &destination)
    {
        int encode_coeff_count = value.significant_bit_count();
        destination.resize(encode_coeff_count);
        destination.set_zero();

        int coeff_index = 0;
        int coeff_count = value.significant_bit_count();
        int coeff_uint64_count = value.uint64_count();
        while (coeff_index < coeff_count)
        {
            if (is_bit_set_uint(value.pointer(), coeff_uint64_count, coeff_index))
            {
                destination[coeff_index] = 1;
            }
            coeff_index++;
        }
    }

    uint32_t BinaryEncoder::decode_uint32(const Plaintext &plain)
    {
        uint64_t value64 = decode_uint64(plain);
        if (value64 > UINT32_MAX)
        {
#ifdef SEAL_THROW_ON_DECODER_OVERFLOW
            throw invalid_argument("output out of range line 120");
#endif
        }
        return static_cast<uint32_t>(value64);
    }

    uint64_t BinaryEncoder::decode_uint64(const Plaintext &plain)
    {
        BigUInt bigvalue = decode_biguint(plain);
        int bit_count = bigvalue.significant_bit_count();
        if (bit_count > bits_per_uint64)
        {
            // Decoded value has more bits than fit in a 64-bit uint.
#ifdef SEAL_THROW_ON_DECODER_OVERFLOW
            throw invalid_argument("output out of range line 133");
#endif
        }
        return bit_count > 0 ? bigvalue.pointer()[0] : 0;
    }

    int32_t BinaryEncoder::decode_int32(const Plaintext &plain)
    {
        int64_t value64 = decode_int64(plain);
        if (value64 < INT32_MIN || value64 > INT32_MAX)
        {
#ifdef SEAL_THROW_ON_DECODER_OVERFLOW
            throw invalid_argument("output out of range line 146");
#endif
        }
        return static_cast<int32_t>(value64);
    }

    int64_t BinaryEncoder::decode_int64(const Plaintext &plain)
    {
        uint64_t pos_value;

        // Determine coefficient threshold for negative numbers.
        int64_t result = 0;
        for (int bit_index = plain.significant_coeff_count() - 1; bit_index >= 0; bit_index--)
        {
            uint64_t coeff = plain[bit_index];
            cout << "bit index " << bit_index << endl;
            cout << "coeff " << coeff << endl;

            // Left shift result.
            int64_t next_result = result << 1;
            if ((next_result < 0) != (result < 0))
            {
                // Check for overflow.
#ifdef SEAL_THROW_ON_DECODER_OVERFLOW
                throw invalid_argument("output out of range line 168");
#endif
            }

            // Get sign/magnitude of coefficient.
            int coeff_bit_count = get_significant_bit_count(coeff);
            if (coeff >= plain_modulus_.value())
            {
                // Coefficient is bigger than plaintext modulus
                throw invalid_argument("plain does not represent a valid plaintext polynomial");
            }
            bool coeff_is_negative = coeff >= coeff_neg_threshold_;
            const uint64_t *pos_pointer;
            if (coeff_is_negative)
            {
                if (sub_uint64(plain_modulus_.value(), coeff, 0, &pos_value))
                {
                    // Check for borrow, which means value is greater than plain_modulus.
                    throw invalid_argument("plain does not represent a valid plaintext polynomial");
                }
                pos_pointer = &pos_value;
                coeff_bit_count = get_significant_bit_count(pos_value);
            }
            else
            {
                pos_pointer = &coeff;
            }
            if (coeff_bit_count > bits_per_uint64 - 1)
            {
                // Absolute value of coefficient is too large to represent in a int64_t, so overflow.
#ifdef SEAL_THROW_ON_DECODER_OVERFLOW
                throw invalid_argument("output out of range line 199");
#endif
            }
            int64_t coeff_value = static_cast<int64_t>(*pos_pointer);
            if (coeff_is_negative)
            {
                coeff_value = -coeff_value;
            }
            bool next_result_was_negative = next_result < 0;
            next_result += coeff_value;
            bool next_result_is_negative = next_result < 0;
            if (next_result_was_negative == coeff_is_negative && next_result_was_negative != next_result_is_negative)
            {
                // Accumulation and coefficient had same signs, but accumulator changed signs after addition, so must be overflow.
#ifdef SEAL_THROW_ON_DECODER_OVERFLOW
                throw invalid_argument("output out of range line 214");
#endif
            }
            result = next_result;
        }
        return result;
    }

    BigUInt BinaryEncoder::decode_biguint(const Plaintext &plain)
    {
        uint64_t pos_value;

        // Determine coefficient threshold for negative numbers.
        int result_uint64_count = 1;
        int result_bit_capacity = result_uint64_count * bits_per_uint64;
        BigUInt resultint(result_bit_capacity);
        bool result_is_negative = false;
        uint64_t *result = resultint.pointer();
        for (int bit_index = plain.significant_coeff_count() - 1; bit_index >= 0; bit_index--)
        {
            uint64_t coeff = plain[bit_index];

            // Left shift result, resizing if highest bit set.
            if (is_bit_set_uint(result, result_uint64_count, result_bit_capacity - 1))
            {
                // Resize to make bigger.
                result_uint64_count++;
                result_bit_capacity = result_uint64_count * bits_per_uint64;
                resultint.resize(result_bit_capacity);
                result = resultint.pointer();
            }
            left_shift_uint(result, 1, result_uint64_count, result);

            // Get sign/magnitude of coefficient.
            if (coeff >= plain_modulus_.value())
            {
                // Coefficient is bigger than plaintext modulus
                throw invalid_argument("plain does not represent a valid plaintext polynomial");
            }
            bool coeff_is_negative = coeff >= coeff_neg_threshold_;
            const uint64_t *pos_pointer;
            if (coeff_is_negative)
            {
                if (sub_uint64(plain_modulus_.value(), coeff, 0, &pos_value))
                {
                    // Check for borrow, which means value is greater than plain_modulus.
                    throw invalid_argument("plain does not represent a valid plaintext polynomial");
                }
                pos_pointer = &pos_value;
            }
            else
            {
                pos_pointer = &coeff;
            }

            // Add or subtract-in coefficient.
            if (result_is_negative == coeff_is_negative)
            {
                // Result and coefficient have same signs so add.
                if (add_uint_uint64(result, *pos_pointer, result_uint64_count, result))
                {
                    // Add produced a carry that didn't fit, so resize and put it in.
                    int carry_bit_index = result_uint64_count * bits_per_uint64;
                    result_uint64_count++;
                    result_bit_capacity = result_uint64_count * bits_per_uint64;
                    resultint.resize(result_bit_capacity);
                    result = resultint.pointer();
                    set_bit_uint(result, result_uint64_count, carry_bit_index);
                }
            }
            else
            {
                // Result and coefficient have opposite signs so subtract.
                if (sub_uint_uint64(result, *pos_pointer, result_uint64_count, result))
                {
                    // Subtraction produced a borrow so coefficient is larger (in magnitude) than result, so need to negate result.
                    negate_uint(result, result_uint64_count, result);
                    result_is_negative = !result_is_negative;
                }
            }
        }

        // Verify result is non-negative.
        if (result_is_negative && !resultint.is_zero())
        {
#ifdef SEAL_THROW_ON_DECODER_OVERFLOW
            throw invalid_argument("poly must decode to positive value");
#endif
        }
        return resultint;
    }

    void BinaryEncoder::decode_biguint(const Plaintext &plain, BigUInt &destination)
    {
        uint64_t pos_value;

        // Determine coefficient threshold for negative numbers.
        destination.set_zero();
        int result_uint64_count = destination.uint64_count();
        int result_bit_capacity = result_uint64_count * bits_per_uint64;
        bool result_is_negative = false;
        uint64_t *result = destination.pointer();
        for (int bit_index = plain.significant_coeff_count() - 1; bit_index >= 0; bit_index--)
        {
            uint64_t coeff = plain[bit_index];

            // Left shift result, failing if highest bit set.
            if (is_bit_set_uint(result, result_uint64_count, result_bit_capacity - 1))
            {
                throw invalid_argument("plain does not represent a valid plaintext polynomial");
            }
            left_shift_uint(result, 1, result_uint64_count, result);

            // Get sign/magnitude of coefficient.
            if (coeff >= plain_modulus_.value())
            {
                // Coefficient is bigger than plaintext modulus.
                throw invalid_argument("plain does not represent a valid plaintext polynomial");
            }
            bool coeff_is_negative = coeff >= coeff_neg_threshold_;
            const uint64_t *pos_pointer;
            if (coeff_is_negative)
            {
                if (sub_uint64(plain_modulus_.value(), coeff, 0, &pos_value))
                {
                    // Check for borrow, which means value is greater than plain_modulus.
                    throw invalid_argument("plain does not represent a valid plaintext polynomial");
                }
                pos_pointer = &pos_value;
            }
            else
            {
                pos_pointer = &coeff;
            }

            // Add or subtract-in coefficient.
            if (result_is_negative == coeff_is_negative)
            {
                // Result and coefficient have same signs so add.
                if (add_uint_uint64(result, *pos_pointer, result_uint64_count, result))
                {
                    // Add produced a carry that didn't fit.
#ifdef SEAL_THROW_ON_DECODER_OVERFLOW
                    throw invalid_argument("output out of range line 357");
#endif
                }
            }
            else
            {
                // Result and coefficient have opposite signs so subtract.
                if (sub_uint_uint64(result, *pos_pointer, result_uint64_count, result))
                {
                    // Subtraction produced a borrow so coefficient is larger (in magnitude) than result, so need to negate result.
                    negate_uint(result, result_uint64_count, result);
                    result_is_negative = !result_is_negative;
                }
            }
        }

        // Verify result is non-negative.
        if (result_is_negative && !destination.is_zero())
        {
#ifdef SEAL_THROW_ON_DECODER_OVERFLOW
            throw invalid_argument("poly must decode to a positive value");
#endif
        }

        // Verify result fits in actual bit-width (as opposed to capacity) of destination.
        if (destination.significant_bit_count() > destination.bit_count())
        {
#ifdef SEAL_THROW_ON_DECODER_OVERFLOW
            throw invalid_argument("output out of range line 385");
#endif
        }
    }

    BalancedEncoder::BalancedEncoder(const SmallModulus &plain_modulus, uint64_t base, const MemoryPoolHandle &pool) :
        pool_(pool),
        plain_modulus_(plain_modulus),
        base_(base),
        coeff_neg_threshold_((plain_modulus.value() + 1) >> 1)
    {
        if (base <= 2)
        {
            throw invalid_argument("base must be at least 3");
        }
        if (*plain_modulus.pointer() < base)
        {
            throw invalid_argument("plain_modulus must be at least b");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }
    }

    Plaintext BalancedEncoder::encode(uint64_t value)
    {
        Plaintext result;
        encode(value, result);

        return result;
    }

    void BalancedEncoder::encode(uint64_t value, Plaintext &destination)
    {
        // We estimate the number of coefficients in the expansion
        int encode_coeff_count = ceil(static_cast<double>(get_significant_bit_count(value)) / log2(base_)) + 1;
        destination.resize(encode_coeff_count);
        destination.set_zero();

        int coeff_index = 0;
        while (value)
        {
            uint64_t remainder = value % base_;
            if (0 < remainder && remainder <= (base_ - 1) / 2)
            {
                destination[coeff_index] = remainder;
            }
            else if (remainder > (base_ - 1) / 2)
            {
                destination[coeff_index] = plain_modulus_.value() - base_ + remainder;
            }
            value = (value + base_ / 2) / base_;

            coeff_index++;
        }
    }

    Plaintext BalancedEncoder::encode(int64_t value)
    {
        Plaintext result;
        encode(value, result);
        return result;
    }

    void BalancedEncoder::encode(int64_t value, Plaintext &destination)
    {
        if (value < 0)
        {
            uint64_t pos_value = static_cast<uint64_t>(-value);

            // We estimate the number of coefficients in the expansion
            int encode_coeff_count = ceil(static_cast<double>(get_significant_bit_count(value)) / log2(base_)) + 1;
            destination.resize(encode_coeff_count);
            destination.set_zero();

            int coeff_index = 0;
            while (pos_value)
            {
                uint64_t remainder = pos_value % base_;
                if (0 < remainder && remainder <= (base_ - 1) / 2)
                {
                    destination[coeff_index] = plain_modulus_.value() - remainder;
                }
                else if (remainder > (base_ - 1) / 2)
                {
                    destination[coeff_index] = base_ - remainder;

                    if ((base_ % 2 == 0) && (remainder == base_ / 2))
                    {
                        destination[coeff_index] = plain_modulus_.value() - destination[coeff_index];
                    }
                }

                // Note that we are adding now (base_-1)/2 instead of base_/2 as in the even case, because value is negative.
                pos_value = (pos_value + ((base_ - 1) / 2)) / base_;

                coeff_index++;
            }
        }
        else
        {
            encode(static_cast<uint64_t>(value), destination);
        }
    }

    Plaintext BalancedEncoder::encode(const BigUInt &value)
    {
        Plaintext result;
        encode(value, result);
        return result;
    }

    void BalancedEncoder::encode(const BigUInt &value, Plaintext &destination)
    {
        if (value.is_zero())
        {
            destination.set_zero();
            return;
        }

        // We estimate the number of coefficients in the expansion
        int encode_coeff_count = ceil(static_cast<double>(value.significant_bit_count()) / log2(base_)) + 1;
        int encode_uint64_count = divide_round_up(encode_coeff_count, bits_per_uint64);

        destination.resize(encode_coeff_count);
        destination.set_zero();

        Pointer base_uint(allocate_uint(encode_uint64_count, pool_));
        set_uint(base_, encode_uint64_count, base_uint.get());
        Pointer base_div_two_uint(allocate_uint(encode_uint64_count, pool_));
        right_shift_uint(base_uint.get(), 1, encode_uint64_count, base_div_two_uint.get());
        uint64_t mod_minus_base = plain_modulus_.value() - base_;

        Pointer quotient(allocate_uint(encode_uint64_count, pool_));
        Pointer remainder(allocate_uint(encode_uint64_count, pool_));
        Pointer temp(allocate_uint(value.uint64_count(), pool_));
        set_uint_uint(value.pointer(), value.uint64_count(), temp.get());

        int coeff_index = 0;
        while (!is_zero_uint(temp.get(), value.uint64_count()))
        {
            divide_uint_uint(temp.get(), base_uint.get(), encode_uint64_count, quotient.get(), remainder.get(), pool_);
            uint64_t *dest_coeff = destination.pointer() + coeff_index;
            if (is_greater_than_uint_uint(remainder.get(), base_div_two_uint.get(), encode_uint64_count))
            {
                *dest_coeff = mod_minus_base + remainder[0];
            }
            else if (!is_zero_uint(remainder.get(), encode_uint64_count))
            {
                *dest_coeff = remainder[0];
            }
            add_uint_uint(temp.get(), base_div_two_uint.get(), encode_uint64_count, temp.get());
            divide_uint_uint(temp.get(), base_uint.get(), encode_uint64_count, quotient.get(), remainder.get(), pool_);
            set_uint_uint(quotient.get(), encode_uint64_count, temp.get());

            coeff_index++;
        }
    }

    uint32_t BalancedEncoder::decode_uint32(const Plaintext &plain)
    {
        uint64_t value64 = decode_uint64(plain);
        if (value64 > UINT32_MAX)
        {
#ifdef SEAL_THROW_ON_DECODER_OVERFLOW
            throw invalid_argument("output out of range line 551");
#endif
        }
        return static_cast<uint32_t>(value64);
    }

    uint64_t BalancedEncoder::decode_uint64(const Plaintext &plain)
    {
        BigUInt bigvalue = decode_biguint(plain);
        int bit_count = bigvalue.significant_bit_count();
        if (bit_count > bits_per_uint64)
        {
            // Decoded value has more bits than fit in a 64-bit uint.
#ifdef SEAL_THROW_ON_DECODER_OVERFLOW
            throw invalid_argument("output out of range line 565");
#endif
        }
        return bit_count > 0 ? bigvalue.pointer()[0] : 0;
    }

    int32_t BalancedEncoder::decode_int32(const Plaintext &plain)
    {
        int64_t value64 = decode_int64(plain);
        if (value64 < INT32_MIN || value64 > INT32_MAX)
        {
#ifdef SEAL_THROW_ON_DECODER_OVERFLOW
            throw invalid_argument("output out of range line 577");
#endif
        }
        return static_cast<int32_t>(value64);
    }

    int64_t BalancedEncoder::decode_int64(const Plaintext &plain)
    {
        uint64_t pos_value;

        // Determine coefficient threshold for negative numbers.
        int64_t result = 0;
        for (int bit_index = plain.significant_coeff_count() - 1; bit_index >= 0; bit_index--)
        {
            uint64_t coeff = plain[bit_index];

            // Multiply result by base.
            int64_t next_result = result * base_;

            if ((next_result < 0) != (result < 0))
            {
                // Check for overflow.
#ifdef SEAL_THROW_ON_DECODER_OVERFLOW
                throw invalid_argument("output out of range line 600");
#endif
            }

            // Get sign/magnitude of coefficient.
            int coeff_bit_count = get_significant_bit_count(coeff);
            if (coeff >= plain_modulus_.value())
            {
                // Coefficient is bigger than plaintext modulus
                throw invalid_argument("plain does not represent a valid plaintext polynomial");
            }
            bool coeff_is_negative = coeff >= coeff_neg_threshold_;
            const uint64_t *pos_pointer;
            if (coeff_is_negative)
            {
                if (sub_uint64(plain_modulus_.value(), coeff, 0, &pos_value))
                {
                    // Check for borrow, which means value is greater than plain_modulus.
                    throw invalid_argument("plain does not represent a valid plaintext polynomial");
                }
                pos_pointer = &pos_value;
                coeff_bit_count = get_significant_bit_count(pos_value);
            }
            else
            {
                pos_pointer = &coeff;
            }
            if (coeff_bit_count > bits_per_uint64 - 1)
            {
                // Absolute value of coefficient is too large to represent in a int64_t, so overflow.
#ifdef SEAL_THROW_ON_DECODER_OVERFLOW
                throw invalid_argument("output out of range line 631");
#endif
            }
            int64_t coeff_value = static_cast<int64_t>(*pos_pointer);
            if (coeff_is_negative)
            {
                coeff_value = -coeff_value;
            }
            bool next_result_was_negative = next_result < 0;
            next_result += coeff_value;
            bool next_result_is_negative = next_result < 0;
            if (next_result_was_negative == coeff_is_negative && next_result_was_negative != next_result_is_negative)
            {
                // Accumulation and coefficient had same signs, but accumulator changed signs after addition, so must be overflow.
#ifdef SEAL_THROW_ON_DECODER_OVERFLOW
                throw invalid_argument("output out of range line 646");
#endif
            }
            result = next_result;
        }
        return result;
    }

    BigUInt BalancedEncoder::decode_biguint(const Plaintext &plain)
    {
        // Determine plain_modulus width.
        uint64_t pos_value;

        // Determine coefficient threshold for negative numbers.
        int result_uint64_count = 1;
        int result_bit_capacity = result_uint64_count * bits_per_uint64;
        BigUInt resultint(result_bit_capacity);
        bool result_is_negative = false;
        uint64_t *result = resultint.pointer();

        BigUInt base_uint(result_bit_capacity);
        base_uint = base_;
        BigUInt temp_result(result_bit_capacity);

        for (int bit_index = plain.significant_coeff_count() - 1; bit_index >= 0; bit_index--)
        {
            uint64_t coeff = plain[bit_index];

            // Multiply result by base. Resize if highest bit set.
            if (is_bit_set_uint(result, result_uint64_count, result_bit_capacity - 1))
            {
                // Resize to make bigger.
                result_uint64_count++;
                result_bit_capacity = result_uint64_count * bits_per_uint64;
                resultint.resize(result_bit_capacity);
                result = resultint.pointer();
            }
            set_uint_uint(result, result_uint64_count, temp_result.pointer());
            multiply_uint_uint(temp_result.pointer(), result_uint64_count, base_uint.pointer(), result_uint64_count, result_uint64_count, result);

            // Get sign/magnitude of coefficient.
            if (coeff >= plain_modulus_.value())
            {
                // Coefficient is bigger than plaintext modulus.
                throw invalid_argument("plain does not represent a valid plaintext polynomial");
            }
            bool coeff_is_negative = coeff >= coeff_neg_threshold_;
            const uint64_t *pos_pointer;
            if (coeff_is_negative)
            {
                if (sub_uint64(plain_modulus_.value(), coeff, 0, &pos_value))
                {
                    // Check for borrow, which means value is greater than plain_modulus.
                    throw invalid_argument("plain does not represent a valid plaintext polynomial");
                }
                pos_pointer = &pos_value;
            }
            else
            {
                pos_pointer = &coeff;
            }

            // Add or subtract-in coefficient.
            if (result_is_negative == coeff_is_negative)
            {
                // Result and coefficient have same signs so add.
                if (add_uint_uint64(result, *pos_pointer, result_uint64_count, result))
                {
                    // Add produced a carry that didn't fit, so resize and put it in.
                    int carry_bit_index = result_uint64_count * bits_per_uint64;
                    result_uint64_count++;
                    result_bit_capacity = result_uint64_count * bits_per_uint64;
                    resultint.resize(result_bit_capacity);
                    result = resultint.pointer();
                    set_bit_uint(result, result_uint64_count, carry_bit_index);
                }
            }
            else
            {
                // Result and coefficient have opposite signs so subtract.
                if (sub_uint_uint64(result, *pos_pointer, result_uint64_count, result))
                {
                    // Subtraction produced a borrow so coefficient is larger (in magnitude) than result, so need to negate result.
                    negate_uint(result, result_uint64_count, result);
                    result_is_negative = !result_is_negative;
                }
            }
        }

        // Verify result is non-negative.
        if (result_is_negative && !resultint.is_zero())
        {
#ifdef SEAL_THROW_ON_DECODER_OVERFLOW
            throw invalid_argument("poly must decode to a positive value");
#endif
        }
        return resultint;
    }

    void BalancedEncoder::decode_biguint(const Plaintext &plain, BigUInt &destination)
    {
        // Determine plain_modulus width.
        uint64_t pos_value;

        // Determine coefficient threshold for negative numbers.
        destination.set_zero();
        int result_uint64_count = destination.uint64_count();
        int result_bit_capacity = result_uint64_count * bits_per_uint64;
        bool result_is_negative = false;
        uint64_t *result = destination.pointer();

        BigUInt base_uint(result_bit_capacity);
        BigUInt temp_result(result_bit_capacity);
        base_uint = base_;

        for (int bit_index = plain.significant_coeff_count() - 1; bit_index >= 0; bit_index--)
        {
            uint64_t coeff = plain[bit_index];

            // Multiply result by base, failing if highest bit set.
            if (is_bit_set_uint(result, result_uint64_count, result_bit_capacity - 1))
            {
                throw invalid_argument("plain does not represent a valid plaintext polynomial");
            }
            set_uint_uint(result, result_uint64_count, temp_result.pointer());
            multiply_truncate_uint_uint(temp_result.pointer(), base_uint.pointer(), result_uint64_count, result);

            // Get sign/magnitude of coefficient.
            if (coeff >= plain_modulus_.value())
            {
                // Coefficient has more bits than plain_modulus.
                throw invalid_argument("plain does not represent a valid plaintext polynomial");
            }
            bool coeff_is_negative = coeff >= coeff_neg_threshold_;
            const uint64_t *pos_pointer;
            if (coeff_is_negative)
            {
                if (sub_uint64(plain_modulus_.value(), coeff, 0, &pos_value))
                {
                    // Check for borrow, which means value is greater than plain_modulus.
                    throw invalid_argument("plain does not represent a valid plaintext polynomial");
                }
                pos_pointer = &pos_value;
            }
            else
            {
                pos_pointer = &coeff;
            }

            // Add or subtract-in coefficient.
            if (result_is_negative == coeff_is_negative)
            {
                // Result and coefficient have same signs so add.
                if (add_uint_uint64(result, *pos_pointer, result_uint64_count, result))
                {
                    // Add produced a carry that didn't fit.
#ifdef SEAL_THROW_ON_DECODER_OVERFLOW
                    throw invalid_argument("output out of range line 803");
#endif
                }
            }
            else
            {
                // Result and coefficient have opposite signs so subtract.
                if (sub_uint_uint64(result, *pos_pointer, result_uint64_count, result))
                {
                    // Subtraction produced a borrow so coefficient is larger (in magnitude) than result, so need to negate result.
                    negate_uint(result, result_uint64_count, result);
                    result_is_negative = !result_is_negative;
                }
            }
        }

        // Verify result is non-negative.
        if (result_is_negative && !destination.is_zero())
        {
#ifdef SEAL_THROW_ON_DECODER_OVERFLOW
            throw invalid_argument("poly must decode to a positive value");
#endif
        }

        // Verify result fits in actual bit-width (as opposed to capacity) of destination.
        if (destination.significant_bit_count() > destination.bit_count())
        {
#ifdef SEAL_THROW_ON_DECODER_OVERFLOW
            throw invalid_argument("output out of range line 831");
#endif
        }
    }

    BinaryFractionalEncoder::BinaryFractionalEncoder(const SmallModulus &plain_modulus, const BigPoly &poly_modulus,
        int integer_coeff_count, int fraction_coeff_count, const MemoryPoolHandle &pool) :
        pool_(pool),
        encoder_(plain_modulus, pool),
        fraction_coeff_count_(fraction_coeff_count),
        integer_coeff_count_(integer_coeff_count),
        poly_modulus_(poly_modulus)
    {
        if (integer_coeff_count <= 0)
        {
            throw invalid_argument("integer_coeff_count must be positive");
        }
        if (fraction_coeff_count <= 0)
        {
            throw invalid_argument("fraction_coeff_count must be positive");
        }
        if (poly_modulus_.is_zero())
        {
            throw invalid_argument("poly_modulus cannot be zero");
        }
        if (integer_coeff_count_ + fraction_coeff_count_ >= poly_modulus_.coeff_count())
        {
            throw invalid_argument("integer/fractional parts are too large for poly_modulus");
        }
        if (poly_modulus_.coeff_count() != poly_modulus_.significant_coeff_count())
        {
            poly_modulus_.resize(poly_modulus_.significant_coeff_count(), poly_modulus.coeff_bit_count());
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }
    }

    Plaintext BinaryFractionalEncoder::encode(double value)
    {
        int coeff_count = poly_modulus_.coeff_count();

        // Take care of the integral part
        int64_t value_int = static_cast<int64_t>(value);
        Plaintext encoded_int(pool_);
        encoder_.encode(value_int, encoded_int);
        value -= value_int;

        // If the fractional part is zero, return encoded_int
        if (value == 0)
        {
            return encoded_int;
        }

        bool is_negative = value < 0;

        //Extract the fractional part
        Plaintext encoded_fract(coeff_count);
        for (int i = 0; i < fraction_coeff_count_; i++)
        {
            value *= 2;
            value_int = static_cast<int64_t>(value);
            value -= value_int;

            // We want to encode the least significant bit of value_int to the least significant bit of encoded_fract.
            // First set it to 1 if it is to be set at all. Later we will negate them all if the number was negative.
            encoded_fract[0] = static_cast<uint64_t>(value_int & 1);

            // Shift encoded_fract by one coefficient unless we are at the last coefficient
            if (i < fraction_coeff_count_ - 1)
            {
                left_shift_uint(encoded_fract.pointer(), bits_per_uint64, coeff_count, encoded_fract.pointer());
            }
        }

        // We negate the coefficients only if the number was NOT negative.
        // This is because the coefficients will have to be negated in any case (sign changes at "wrapping around"
        // the polynomial modulus).
        if (!is_negative)
        {
            for (int i = 0; i < fraction_coeff_count_; i++)
            {
                if (encoded_fract[i] != 0)
                {
                    encoded_fract[i] = encoder_.neg_one_;
                }
            }
        }

        // Shift the fractional part to top of polynomial
        left_shift_uint(encoded_fract.pointer(), bits_per_uint64 * (coeff_count - 1 - fraction_coeff_count_),
            coeff_count, encoded_fract.pointer());

        // Combine everything together
        set_uint_uint(encoded_int.pointer(), encoded_int.coeff_count(), encoded_fract.pointer());

        return encoded_fract;
    }

    double BinaryFractionalEncoder::decode(const Plaintext &plain)
    {
        int coeff_count = poly_modulus_.coeff_count();

        // Validate input parameters
        if (plain.coeff_count() > coeff_count || (plain.coeff_count() == coeff_count && plain[coeff_count - 1] != 0))
        {
            throw invalid_argument("plain is not valid for encryption parameters");
        }
#ifdef SEAL_DEBUG
        if (plain.significant_coeff_count() >= coeff_count || !are_poly_coefficients_less_than(plain.pointer(),
            plain.coeff_count(), 1, encoder_.plain_modulus_.pointer(), encoder_.plain_modulus().uint64_count()))
        {
            throw invalid_argument("plain is not valid for encryption parameters");
        }
#endif
        // Do we have an empty plaintext
        if (plain.coeff_count() == 0)
        {
            return 0;
        }

        // plain might be smaller than expected if leading coefficients are missing
        Pointer plain_copy(allocate_zero_uint(coeff_count, pool_));
        set_uint_uint(plain.pointer(), plain.coeff_count(), plain_copy.get());

        // Extract the fractional and integral parts
        Plaintext encoded_int(integer_coeff_count_, pool_);
        Pointer encoded_fract(allocate_zero_uint(fraction_coeff_count_, pool_));

        // Integer part
        set_uint_uint(plain_copy.get(), integer_coeff_count_, encoded_int.pointer());

        // Read from the top of the poly all the way to the top of the integral part to obtain the fractional part
        set_uint_uint(plain_copy.get() + coeff_count - 1 - fraction_coeff_count_, fraction_coeff_count_, encoded_fract.get());

        // Decode integral part
        cout << "Decoding BinaryFractionalEncoder int part" << endl;
        int64_t integral_part = encoder_.decode_int64(encoded_int);
        cout << "Decoded BinaryFractionalEncoder int part " << integral_part << endl;

        // Decode fractional part (or rather negative of it), one coefficient at a time
        double fractional_part = 0;
        Plaintext temp_int_part(1, pool_);
        for (int i = 0; i < fraction_coeff_count_; i++)
        {
            cout << i << " ";
            temp_int_part[0] = encoded_fract[i];
            //cout << "Decing BinaryFractionalEncoder frac part " << i << " of " << fraction_coeff_count_ << endl;
            fractional_part += encoder_.decode_int64(temp_int_part);
            //cout << "deoded i";
            fractional_part /= 2;
        }

        return static_cast<double>(integral_part) - fractional_part;
    }

    BalancedFractionalEncoder::BalancedFractionalEncoder(const SmallModulus &plain_modulus, const BigPoly &poly_modulus, int integer_coeff_count, int fraction_coeff_count, uint64_t base, const MemoryPoolHandle &pool) :
        pool_(pool),
        encoder_(plain_modulus, base, pool_),
        fraction_coeff_count_(fraction_coeff_count),
        integer_coeff_count_(integer_coeff_count),
        poly_modulus_(poly_modulus)
    {
        if (integer_coeff_count <= 0)
        {
            throw invalid_argument("integer_coeff_count must be positive");
        }
        if (fraction_coeff_count <= 0)
        {
            throw invalid_argument("fraction_coeff_count must be positive");
        }
        if (poly_modulus_.is_zero())
        {
            throw invalid_argument("poly_modulus cannot be zero");
        }
        if (integer_coeff_count_ + fraction_coeff_count_ >= poly_modulus_.coeff_count())
        {
            throw invalid_argument("integer/fractional parts are too large for poly_modulus");
        }
        if (poly_modulus_.coeff_count() != poly_modulus_.significant_coeff_count())
        {
            poly_modulus_.resize(poly_modulus_.significant_coeff_count(), poly_modulus.coeff_bit_count());
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }
    }

    // We encode differently based on whether the base is odd or even.
    Plaintext BalancedFractionalEncoder::encode(double value)
    {
        if (encoder_.base_ & 1)
        {
            return encode_odd(value);
        }
        else
        {
            return encode_even(value);
        }
    }

    Plaintext BalancedFractionalEncoder::encode_odd(double value)
    {
        int coeff_count = poly_modulus_.coeff_count();

        // Take care of the integral part
        int64_t value_int = static_cast<int64_t>(round(value));
        Plaintext encoded_int(pool_);
        encoder_.encode(value_int, encoded_int);
        value -= value_int;

        // If the fractional part is zero, return encoded_int
        if (value == 0)
        {
            return encoded_int;
        }

        // Extract the fractional part
        Plaintext encoded_fract(coeff_count);
        for (int i = 0; i < fraction_coeff_count_; i++)
        {
            value *= encoder_.base();

            // When computing the next value_int we need to round e.g. 0.5 to 0 (not to 1) and
            // -0.5 to 0 (not to -1), i.e. always towards zero.
            int sign = (value >= 0 ? 1 : -1);
            value_int = static_cast<int64_t>(sign * ceil(abs(value) - 0.5));
            value -= value_int;

            // We store the representative of value_int modulo the base (symmetric representative)
            // as the absolute value (in value_int_mod_base) and as the sign (in is_negative).
            bool is_negative = false;

            if (value_int < 0)
            {
                is_negative = true;
                value_int = -value_int;
            }

            // Set the constant coefficient of encoded_fract to be the correct absolute value.
            encoded_fract[0] = value_int;
            // And negate it modulo plain_modulus if it was NOT supposed to be negative, because the
            // fractional encoding requires the signs of the fractional coefficients to be negatives of
            // what one might naively expect, as they change sign when "wrapping around" the polynomial modulus.
            if (!is_negative && value_int != 0)
            {
                encoded_fract[0] = encoder_.plain_modulus_.value() - encoded_fract[0];
            }

            // Shift encoded_fract by one coefficient unless we are at the last coefficient
            if (i < fraction_coeff_count_ - 1)
            {
                left_shift_uint(encoded_fract.pointer(), bits_per_uint64, coeff_count, encoded_fract.pointer());
            }
        }

        // Shift the fractional part to top of polynomial
        left_shift_uint(encoded_fract.pointer(), bits_per_uint64 * (coeff_count - 1 - fraction_coeff_count_),
            coeff_count, encoded_fract.pointer());

        // Combine everything together
        set_uint_uint(encoded_int.pointer(), encoded_int.coeff_count(), encoded_fract.pointer());

        return encoded_fract;
    }

    Plaintext BalancedFractionalEncoder::encode_even(double value)
    {
        int coeff_count = poly_modulus_.coeff_count();

        // Take care of the integral part
        int64_t value_int = static_cast<int64_t>(round(value));

        // We store the integral part for further use, since we may end up changing the integral part based on our encoding of the fractional part
        int64_t initial = value_int;

        Plaintext encoded_int(coeff_count, pool_);
        encoder_.encode(value_int, encoded_int);
        value -= value_int;

        // If the fractional part is zero, return encoded_int
        if (value == 0)
        {
            return encoded_int;
        }

        // Extract the fractional part
        // We will first compute the balanced base b encoding of the fractional part, allowing coefficients in the range -b/2, ..., b/2
        // We use Pointer carry to mark the coefficients that are equal to b/2, and we use Pointer is_less_than_neg_one to mark the
        // coefficients that are less than -1 (we need this because when we encounter a coefficient greater than or equal to b/2, we need
        // to store base - coefficient instead and add 1 to the coefficient to the left, which might change the sign of the coefficient
        // to the left).

        Plaintext encoded_fract(coeff_count);
        Pointer carry(allocate_zero_uint(coeff_count, pool_));
        Pointer is_less_than_neg_one(allocate_zero_uint(coeff_count, pool_));
        Pointer is_negative(allocate_zero_uint(coeff_count, pool_));

        for (int i = 0; i < fraction_coeff_count_; i++)
        {
            value *= encoder_.base();

            // When computing the next value_int we need to round e.g. 0.5 to 0 (not to 1) and
            // -0.5 to 0 (not to -1), i.e. always towards zero.
            int sign = (value >= 0 ? 1 : -1);
            value_int = static_cast<int64_t>(sign * ceil(abs(value) - 0.5));
            value -= value_int;

            // Set the constant coefficients of carry, is_less_than_neg_one, is_negative and encoded_fract to be the correct values.
            if ((static_cast<uint64_t>(abs(value_int)) >= encoder_.base_ / 2) && (value_int >= 0))
            {
                carry[0] = 1ULL;
            }
            if (value_int < -1)
            {
                is_less_than_neg_one[0] = 1ULL;
            }
            if (value_int < 0)
            {
                is_negative[0] = 1ULL;
                value_int = -value_int;
            }

            // Set the constant coefficient of encoded_fract to be the correct absolute value.
            encoded_fract[0] = static_cast<uint64_t>(value_int);

            // Shift all the polynomials by one coefficient unless we are at the last coefficient
            if (i < fraction_coeff_count_ - 1)
            {
                left_shift_uint(encoded_fract.pointer(), bits_per_uint64, coeff_count, encoded_fract.pointer());
                left_shift_uint(carry.get(), bits_per_uint64, coeff_count, carry.get());
                left_shift_uint(is_less_than_neg_one.get(), bits_per_uint64, coeff_count, is_less_than_neg_one.get());
                left_shift_uint(is_negative.get(), bits_per_uint64, coeff_count, is_negative.get());
            }
        }

        uint64_t *encoded_fract_ptr = encoded_fract.pointer();
        uint64_t *is_negative_ptr = is_negative.get();
        uint64_t base_div_two = encoder_.base_ / 2;

        // Now we get rid of those coefficients that are greater than or equal to base / 2
        for (int i = 0; i < fraction_coeff_count_ - 1; i++)
        {
            if (carry[i] != 0)
            {
                // Set the sign of the current coefficient to be negative
                is_negative[i] = 1ULL;

                // Store base - current coefficient
                *encoded_fract_ptr = encoder_.base_ - *encoded_fract_ptr;

                // Add 1 to the coefficient to the left. Update the carry entry for the coefficient to the left.
                if (is_negative[i + 1] == 0)
                {
                    encoded_fract_ptr[1]++;
                }
                else
                {
                    encoded_fract_ptr[1]--;

                    // Update the sign of the coefficient to the left if needed
                    if (!is_less_than_neg_one[i + 1])
                    {
                        is_negative[i + 1] = 0;
                    }
                }

                if (encoded_fract_ptr[1] >= base_div_two)
                // if (is_greater_than_or_equal_uint_uint(encoded_fract_ptr + plain_uint64_count, &base_div_two, plain_uint64_count))
                {
                    carry[i + 1] = 1ULL;
                }
            }

            encoded_fract_ptr++;
            is_negative_ptr++;
        }

        // Do we need to change the integral part?
        bool change_int = (carry[fraction_coeff_count_ - 1] != 0);
        if (change_int)
        {
            *encoded_fract_ptr = encoder_.base_ - *encoded_fract_ptr;
            is_negative[fraction_coeff_count_ - 1] = 1ULL;
        }

        // And negate it modulo plain_modulus if it was NOT supposed to be negative, because the
        // fractional encoding requires the signs of the fractional coefficients to be negatives of
        // what one might naively expect, as they change sign when "wrapping around" the polynomial modulus.
        for (int i = fraction_coeff_count_ - 1; i >= 0; --i)
        {
            if ((!is_negative[i]) && (encoded_fract[i] != 0))
            {
                encoded_fract_ptr[0] = encoder_.plain_modulus_.value() - encoded_fract_ptr[0];
            }
            encoded_fract_ptr--;
        }

        // Shift the fractional part to top of polynomial
        left_shift_uint(encoded_fract.pointer(), bits_per_uint64 * (coeff_count - 1 - fraction_coeff_count_),
            coeff_count, encoded_fract.pointer());

        // If change_int is true, then we need to add 1 to the integral part and re-encode it.
        if (change_int)
        {
            encoder_.encode(initial + 1, encoded_int);
        }

        // Combine everything together
        set_uint_uint(encoded_int.pointer(), encoded_int.coeff_count(), encoded_fract.pointer());

        return encoded_fract;
    }

    double BalancedFractionalEncoder::decode(const Plaintext &plain)
    {
        int coeff_count = poly_modulus_.coeff_count();

        // Validate input parameters
        if (plain.coeff_count() > coeff_count || (plain.coeff_count() == coeff_count && plain[coeff_count - 1] != 0))
        {
            throw invalid_argument("plain is not valid for encryption parameters");
        }
#ifdef SEAL_DEBUG
        if (plain.significant_coeff_count() >= coeff_count || !are_poly_coefficients_less_than(plain.pointer(),
            plain.coeff_count(), 1, encoder_.plain_modulus_.pointer(), encoder_.plain_modulus_.uint64_count()))
        {
            throw invalid_argument("plain is not valid for encryption parameters");
        }
#endif
        // plain might be smaller than expected if leading coefficients are missing
        Pointer plain_copy(allocate_zero_uint(coeff_count, pool_));
        set_uint_uint(plain.pointer(), plain.coeff_count(), plain_copy.get());

        // Extract the fractional and integral parts
        Plaintext encoded_int(integer_coeff_count_, pool_);
        Pointer encoded_fract(allocate_zero_uint(fraction_coeff_count_, pool_));

        // Integer part
        set_uint_uint(plain_copy.get(), integer_coeff_count_, encoded_int.pointer());

        // Read from the top of the poly all the way to the top of the integral part to obtain the fractional part
        set_uint_uint(plain_copy.get() + coeff_count - 1 - fraction_coeff_count_, fraction_coeff_count_, encoded_fract.get());

        // Decode integral part
        cout << "Decoding integer part" << endl;
        int64_t integral_part = encoder_.decode_int64(encoded_int);
        cout << "Decoded integer part: " << integral_part << endl;

        // Decode fractional part (or rather negative of it), one coefficient at a time
        double fractional_part = 0;
        Plaintext temp_int_part(1, pool_);
        for (int i = 0; i < fraction_coeff_count_; i++)
        {
            temp_int_part[0] = encoded_fract[i];
            cout << "Decoding fractional part " << i << " of " << fraction_coeff_count_ << endl;
            fractional_part += encoder_.decode_int64(temp_int_part);
            cout << "decoded fractional part" << endl;
            fractional_part /= encoder_.base();
        }

        return static_cast<double>(integral_part) - fractional_part;
    }

    IntegerEncoder::IntegerEncoder(const SmallModulus &plain_modulus, uint64_t base, const MemoryPoolHandle &pool)
    {
        if (base == 2)
        {
            encoder_ = new BinaryEncoder(plain_modulus, pool);
        }
        else
        {
            encoder_ = new BalancedEncoder(plain_modulus, base, pool);
        }
    }

    IntegerEncoder::IntegerEncoder(const IntegerEncoder &copy)
    {
        if (copy.base() == 2)
        {
            encoder_ = new BinaryEncoder(*reinterpret_cast<BinaryEncoder*>(copy.encoder_));
        }
        else
        {
            encoder_ = new BalancedEncoder(*reinterpret_cast<BalancedEncoder*>(copy.encoder_));
        }
    }

    IntegerEncoder::~IntegerEncoder()
    {
        if (encoder_ != nullptr)
        {
            delete encoder_;
            encoder_ = nullptr;
        }
    }

    void IntegerEncoder::encode(uint64_t value, Plaintext &destination)
    {
        encoder_->encode(value, destination);

        // Resize to correct size
        destination.resize(destination.significant_coeff_count());
    }

    void IntegerEncoder::encode(int64_t value, Plaintext &destination)
    {
        encoder_->encode(value, destination);

        // Resize to correct size
        destination.resize(destination.significant_coeff_count());
    }

    void IntegerEncoder::encode(const BigUInt &value, Plaintext &destination)
    {
        encoder_->encode(value, destination);

        // Resize to correct size
        destination.resize(destination.significant_coeff_count());
    }

    void IntegerEncoder::encode(int32_t value, Plaintext &destination)
    {
        encoder_->encode(value, destination);

        // Resize to correct size
        destination.resize(destination.significant_coeff_count());
    }

    void IntegerEncoder::encode(uint32_t value, Plaintext &destination)
    {
        encoder_->encode(value, destination);

        // Resize to correct size
        destination.resize(destination.significant_coeff_count());
    }

    FractionalEncoder::FractionalEncoder(const SmallModulus &plain_modulus, const BigPoly &poly_modulus, int integer_coeff_count, int fraction_coeff_count, uint64_t base, const MemoryPoolHandle &pool)
    {
        if (base == 2)
        {
            encoder_ = new BinaryFractionalEncoder(plain_modulus, poly_modulus, integer_coeff_count, fraction_coeff_count, pool);
        }
        else
        {
            encoder_ = new BalancedFractionalEncoder(plain_modulus, poly_modulus, integer_coeff_count, fraction_coeff_count, base, pool);
        }
    }

    FractionalEncoder::FractionalEncoder(const FractionalEncoder &copy)
    {
        if (copy.base() == 2)
        {
            encoder_ = new BinaryFractionalEncoder(*reinterpret_cast<BinaryFractionalEncoder*>(copy.encoder_));
        }
        else
        {
            encoder_ = new BalancedFractionalEncoder(*reinterpret_cast<BalancedFractionalEncoder*>(copy.encoder_));
        }
    }

    FractionalEncoder::~FractionalEncoder()
    {
        if (encoder_ != nullptr)
        {
            delete encoder_;
            encoder_ = nullptr;
        }
    }
}

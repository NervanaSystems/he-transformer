#include <algorithm>
#include <cstring>
#include <sstream>
#include <limits>
#include "seal/bigpoly.h"
#include "seal/util/common.h"
#include "seal/util/uintcore.h"
#include "seal/util/polycore.h"

using namespace std;
using namespace seal::util;

namespace seal
{
    namespace
    {
        bool is_dec_char(char c)
        {
            return c >= '0' && c <= '9';
        }

        int get_dec_value(char c)
        {
            return c - '0';
        }

        int get_coeff_length(const char *poly)
        {
            int length = 0;
            while (is_hex_char(*poly))
            {
                length++;
                poly++;
            }
            return length;
        }

        int get_coeff_power(const char *poly, int *power_length)
        {
            int length = 0;
            if (*poly == '\0')
            {
                *power_length = 0;
                return 0;
            }
            if (*poly != 'x')
            {
                return -1;
            }
            poly++;
            length++;

            if (*poly != '^')
            {
                return -1;
            }
            poly++;
            length++;

            int power = 0;
            while (is_dec_char(*poly))
            {
                power *= 10;
                power += get_dec_value(*poly);
                poly++;
                length++;
            }
            *power_length = length;
            return power;
        }

        int get_plus(const char *poly)
        {
            if (*poly == '\0')
            {
                return 0;
            }
            if (*poly++ != ' ')
            {
                return -1;
            }
            if (*poly++ != '+')
            {
                return -1;
            }
            if (*poly != ' ')
            {
                return -1;
            }
            return 3;
        }
    }

    BigPoly::BigPoly(int coeff_count, int coeff_bit_count)
    {
        resize(coeff_count, coeff_bit_count);
    }

    BigPoly::BigPoly(const string &hex_poly)
    {
        operator =(hex_poly);
    }

    BigPoly::BigPoly(int coeff_count, int coeff_bit_count, const string &hex_poly)
    {
        resize(coeff_count, coeff_bit_count);
        operator =(hex_poly);
        if (coeff_count_ != coeff_count || coeff_bit_count_ != coeff_bit_count)
        {
            resize(coeff_count, coeff_bit_count);
        }
    }

    BigPoly::BigPoly(int coeff_count, int coeff_bit_count, uint64_t *value)
    {
        alias(coeff_count, coeff_bit_count, value);
    }

    BigPoly::BigPoly(const BigPoly& copy)
    {
        resize(copy.coeff_count(), copy.coeff_bit_count());
        operator =(copy);
    }

    BigPoly::BigPoly(BigPoly &&source) noexcept :
        pool_(move(source.pool_)),
        value_(move(source.value_)),
        coeffs_(move(source.coeffs_)),
        coeff_count_(source.coeff_count_),
        coeff_bit_count_(source.coeff_bit_count_),
        coeff_uint64_count_(source.coeff_uint64_count_),
        is_alias_(source.is_alias_)
    {
        // Pointers in source have been taken over so set them to nullptr
        source.coeffs_.clear();
        source.is_alias_ = false;
        source.coeff_count_ = 0;
        source.coeff_bit_count_ = 0;
        source.coeff_uint64_count_ = 0;
    }

    BigPoly::~BigPoly()
    {
        reset();
    }

    int BigPoly::significant_coeff_bit_count() const
    {
        if (coeff_count_ == 0 || coeff_bit_count_ == 0)
        {
            return 0;
        }
        int max_coeff_sig_bit_count = 0;
        for (int i = 0; i < coeff_count_; i++)
        {
            const BigUInt &coeff = operator[](i);
            int coeff_sig_bit_count = coeff.significant_bit_count();
            if (coeff_sig_bit_count > max_coeff_sig_bit_count)
            {
                max_coeff_sig_bit_count = coeff_sig_bit_count;
            }
        }
        return max_coeff_sig_bit_count;
    }

    string BigPoly::to_string() const
    {
        return poly_to_hex_string(value_.get(), coeff_count_, coeff_uint64_count_);
    }

    void BigPoly::resize(int coeff_count, int coeff_bit_count)
    {
        if (coeff_count < 0)
        {
            throw invalid_argument("coeff_count must be non-negative");
        }
        if (coeff_bit_count < 0)
        {
            throw invalid_argument("coeff_bit_count must be non-negative");
        }
        if (is_alias_)
        {
            throw logic_error("cannot resize an aliased BigPoly");
        }
        if (coeff_count == coeff_count_ && coeff_bit_count == coeff_bit_count_)
        {
            return;
        }

        // Lazy initialization of MemoryPoolHandle
        if (!pool_)
        {
            pool_ = MemoryPoolHandle::Global();
        }

        int coeff_uint64_count = divide_round_up(coeff_bit_count, bits_per_uint64);

        // No resizing needed
        if (coeff_uint64_count_ == coeff_uint64_count && coeff_count == coeff_count_)
        {
            uint64_t *coeff = value_.get();
            for (int i = 0; i < coeff_count; i++)
            {
                filter_highbits_uint(coeff, coeff_uint64_count, coeff_bit_count);
                coeffs_[i].alias(coeff_bit_count, coeff);
                coeff += coeff_uint64_count;
            }
            coeff_bit_count_ = coeff_bit_count;
            return;
        }

        // Allocate new space.
        int uint64_count = coeff_count * coeff_uint64_count;
        Pointer new_value;
        if (uint64_count > 0)
        {
            new_value.swap_with(allocate_uint(uint64_count, pool_));
        }

        // Copy over old values.
        if (uint64_count > 0)
        {
            const uint64_t *from_coeff = value_.get();
            uint64_t *to_coeff = new_value.get();
            for (int i = 0; i < coeff_count; i++)
            {
                if (i < coeff_count_)
                {
                    set_uint_uint(from_coeff, coeff_uint64_count_, coeff_uint64_count, to_coeff);
                    filter_highbits_uint(to_coeff, coeff_uint64_count, coeff_bit_count);
                }
                else
                {
                    set_zero_uint(coeff_uint64_count, to_coeff);
                }
                from_coeff += coeff_uint64_count_;
                to_coeff += coeff_uint64_count;
            }
        }

        // Deallocate any owned pointers.
        //reset();

        // Create coefficients.
        if (coeff_count > 0)
        {
            coeffs_.reserve(coeff_count);
            coeffs_.resize(min<int>(coeff_count, coeffs_.size()));
            uint64_t *new_coeff = new_value.get();

            for (int i = 0; i < coeffs_.size(); i++)
            {
                coeffs_[i].alias(coeff_bit_count, new_coeff);
                new_coeff += coeff_uint64_count;
            }
            for (int i = coeffs_.size(); i < coeff_count; i++)
            {
                coeffs_.emplace_back(coeff_bit_count, new_coeff);
                new_coeff += coeff_uint64_count;
            }
        }

        // Update class.
        value_.swap_with(new_value);
        coeff_count_ = coeff_count;
        coeff_bit_count_ = coeff_bit_count;
        coeff_uint64_count_ = coeff_uint64_count;
        is_alias_ = false;
    }

    void BigPoly::alias(int coeff_count, int coeff_bit_count, uint64_t *value)
    {
        if (coeff_count < 0)
        {
            throw invalid_argument("coeff_count must be non-negative");
        }
        if (coeff_bit_count < 0)
        {
            throw invalid_argument("coeff_bit_count must be non-negative");
        }
        if (value == nullptr && (coeff_count > 0 || coeff_bit_count > 0))
        {
            throw invalid_argument("value must be non-null for non-zero coefficient and bit counts");
        }

        // Deallocate any owned pointers.
        reset();

        // Initialize class.
        value_ = Pointer::Aliasing(value);
        coeff_count_ = coeff_count;
        coeff_bit_count_ = coeff_bit_count;
        coeff_uint64_count_ = divide_round_up(coeff_bit_count, bits_per_uint64);;
        is_alias_ = true;

        // Create coefficients.
        if (coeff_count_ > 0)
        {
            coeffs_.resize(coeff_count);
            uint64_t *new_coeff = value_.get();
            for (int i = 0; i < coeff_count_; i++)
            {
                coeffs_[i].alias(coeff_bit_count_, new_coeff);
                new_coeff += coeff_uint64_count_;
            }
        }
    }

    BigPoly &BigPoly::operator =(const BigPoly &assign)
    {
        // Do nothing if same thing.
        if (&assign == this)
        {
            return *this;
        }

        // Verify assigned polynomial will fit within coefficient and bit counts.
        int assign_sig_coeff_count = assign.significant_coeff_count();

        int assign_max_coeff_bit_count = 0;
        for (int i = 0; i < assign_sig_coeff_count; i++)
        {
            int assign_coeff_bit_count = assign[i].significant_bit_count();
            if (assign_coeff_bit_count > assign_max_coeff_bit_count)
            {
                assign_max_coeff_bit_count = assign_coeff_bit_count;
            }
        }
        if (coeff_count_ < assign_sig_coeff_count || coeff_bit_count_ < assign_max_coeff_bit_count)
        {
            resize(max(assign_sig_coeff_count, coeff_count_), max(assign_max_coeff_bit_count, coeff_bit_count_));
        }

        // Copy it over.
        if (coeff_count_ > 0)
        {
            for (int i = 0; i < coeff_count_; i++)
            {
                BigUInt &coeff = operator[](i);
                if (i < assign_sig_coeff_count)
                {
                    coeff = assign[i];
                }
                else
                {
                    coeff.set_zero();
                }
            }
        }
        return *this;
    }

    BigPoly &BigPoly::operator =(const string &hex_poly)
    {
        int length = hex_poly.size();

        // Determine size needed to store string coefficient.
        int assign_coeff_count = 0;
        int assign_coeff_bit_count = 0;
        int pos = 0;
        int last_power = numeric_limits<int>::max();
        const char *hex_poly_ptr = hex_poly.data();
        while (pos < length)
        {
            // Determine length of coefficient starting at pos.
            int coeff_length = get_coeff_length(hex_poly_ptr + pos);
            if (coeff_length == 0)
            {
                throw invalid_argument("unable to parse hex_poly");
            }

            // Determine bit length of coefficient.
            int coeff_bit_count = get_hex_string_bit_count(hex_poly_ptr + pos, coeff_length);
            if (coeff_bit_count > assign_coeff_bit_count)
            {
                assign_coeff_bit_count = coeff_bit_count;
            }
            pos += coeff_length;

            // Extract power-term.
            int power_length = 0;
            int power = get_coeff_power(hex_poly_ptr + pos, &power_length);
            if (power == -1 || power >= last_power)
            {
                throw invalid_argument("unable to parse hex_poly");
            }
            if (assign_coeff_count == 0)
            {
                assign_coeff_count = power + 1;
            }
            pos += power_length;
            last_power = power;

            // Extract plus (unless it is the end).
            int plus_length = get_plus(hex_poly_ptr + pos);
            if (plus_length == -1)
            {
                throw invalid_argument("unable to parse hex_poly");
            }
            pos += plus_length;
        }

        // If string is empty, then done.
        if (assign_coeff_count == 0 || assign_coeff_bit_count == 0)
        {
            set_zero();
            return *this;
        }

        // Resize polynomial if needed.
        if (coeff_count_ < assign_coeff_count || coeff_bit_count_ < assign_coeff_bit_count)
        {
            resize(max(assign_coeff_count, coeff_count_), max(assign_coeff_bit_count, coeff_bit_count_));
        }

        // Populate polynomial from string.
        int coeff_uint64_count = divide_round_up(coeff_bit_count_, bits_per_uint64);
        pos = 0;
        last_power = coeff_count_;
        while (pos < length)
        {
            // Determine length of coefficient starting at pos.
            const char *coeff_start = hex_poly_ptr + pos;
            int coeff_length = get_coeff_length(coeff_start);
            pos += coeff_length;

            // Extract power-term.
            int power_length = 0;
            int power = get_coeff_power(hex_poly_ptr + pos, &power_length);
            pos += power_length;

            // Extract plus (unless it is the end).
            int plus_length = get_plus(hex_poly_ptr + pos);
            pos += plus_length;

            // Zero coefficients not set by string.
            for (int zero_power = last_power - 1; zero_power > power; --zero_power)
            {
                uint64_t *coeff_ptr = get_poly_coeff(value_.get(), zero_power, coeff_uint64_count);
                set_zero_uint(coeff_uint64_count, coeff_ptr);
            }

            // Populate coefficient.
            uint64_t *coeff_ptr = get_poly_coeff(value_.get(), power, coeff_uint64_count);
            hex_string_to_uint(coeff_start, coeff_length, coeff_uint64_count, coeff_ptr);
            last_power = power;
        }

        // Zero coefficients not set by string.
        for (int zero_power = last_power - 1; zero_power >= 0; --zero_power)
        {
            uint64_t *coeff_ptr = get_poly_coeff(value_.get(), zero_power, coeff_uint64_count);
            set_zero_uint(coeff_uint64_count, coeff_ptr);
        }

        return *this;
    }

    void BigPoly::save(ostream &stream) const
    {
        int32_t coeff_count32 = static_cast<int32_t>(coeff_count_);
        stream.write(reinterpret_cast<const char*>(&coeff_count32), sizeof(int32_t));
        int32_t coeff_bit_count32 = static_cast<int32_t>(coeff_bit_count_);
        stream.write(reinterpret_cast<const char*>(&coeff_bit_count32), sizeof(int32_t));
        int coeff_uint64_count = divide_round_up(coeff_bit_count_, bits_per_uint64);
        stream.write(reinterpret_cast<const char*>(value_.get()), coeff_count_ * coeff_uint64_count * bytes_per_uint64);
    }

    void BigPoly::load(istream &stream)
    {
        int32_t read_coeff_count = 0;
        stream.read(reinterpret_cast<char*>(&read_coeff_count), sizeof(int32_t));
        int32_t read_coeff_bit_count = 0;
        stream.read(reinterpret_cast<char*>(&read_coeff_bit_count), sizeof(int32_t));
        if (read_coeff_count > coeff_count_ || read_coeff_bit_count > coeff_bit_count_)
        {
            // Size is too large to currently fit, so resize.
            resize(max(read_coeff_count, coeff_count_), max(read_coeff_bit_count, coeff_bit_count_));
        }
        int read_coeff_uint64_count = divide_round_up(read_coeff_bit_count, bits_per_uint64);
        int coeff_uint64_count = divide_round_up(coeff_bit_count_, bits_per_uint64);
        if (read_coeff_uint64_count == coeff_uint64_count)
        {
            stream.read(reinterpret_cast<char*>(value_.get()), read_coeff_count * coeff_uint64_count * bytes_per_uint64);
        }
        else
        {
            // Coefficients are different sizes, so read one at a time.
            uint64_t *coeff_ptr = value_.get();
            for (int i = 0; i < read_coeff_count; i++)
            {
                stream.read(reinterpret_cast<char*>(coeff_ptr), read_coeff_uint64_count * bytes_per_uint64);
                set_zero_uint(coeff_uint64_count - read_coeff_uint64_count, coeff_ptr + read_coeff_uint64_count);
                coeff_ptr += coeff_uint64_count;
            }
        }

        // Zero any remaining coefficients.
        if (coeff_count_ > read_coeff_count)
        {
            set_zero_poly(coeff_count_ - read_coeff_count, coeff_uint64_count, value_.get() + read_coeff_count * coeff_uint64_count);
        }
    }
}

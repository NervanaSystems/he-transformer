#pragma once

#include <stdexcept>
#include <cstdint>
#include <vector>
#include <tuple>

namespace seal
{
    namespace util
    {
        inline std::uint64_t gcd(std::uint64_t x, std::uint64_t y)
        {
#ifdef SEAL_DEBUG
            if (x == 0)
            {
                std::invalid_argument("x cannot be zero");
            }
            if (y == 0)
            {
                std::invalid_argument("y cannot be zero");
            }
#endif
            if (x < y)
            {
                return gcd(y, x);
            }
            if (y == 0)
            {
                return x;
            }
            std::uint64_t f = x % y;
            if (f == 0)
            {
                return y;
            }
            else
            {
                return gcd(y, f);
            }
        }

        inline std::tuple<std::uint64_t, std::int64_t, std::int64_t> xgcd(std::uint64_t x, std::uint64_t y)
        {
            /* Extended GCD:
            Returns (gcd, x, y) where gcd is the greatest common divisor of a and b.
            The numbers x, y are such that gcd = ax + by.
            */
#ifdef SEAL_DEBUG
            if (x == 0)
            {
                std::invalid_argument("x cannot be zero");
            }
            if (y == 0)
            {
                std::invalid_argument("y cannot be zero");
            }
#endif
            std::int64_t prev_a = 1;
            std::int64_t a = 0;
            std::int64_t prev_b = 0;
            std::int64_t b = 1;

            while (y != 0)
            {
                std::int64_t q = x / y;
                std::int64_t temp = x % y;
                x = y;
                y = temp;

                temp = a;
                a = prev_a - q * a;
                prev_a = temp;

                temp = b;
                b = prev_b - q * b;
                prev_b = temp;
            }
            return std::make_tuple(x, prev_a, prev_b);
        }

        inline bool try_mod_inverse(std::uint64_t value, std::uint64_t modulus, std::uint64_t &result)
        {
#ifdef SEAL_DEBUG
            if (value == 0)
            {
                std::invalid_argument("value cannot be zero");
            }
            if (modulus <= 1)
            {
                std::invalid_argument("modulus must be at least 2");
            }
            
#endif
            auto gcd_tuple = xgcd(value, modulus);
            if (std::get<0>(gcd_tuple) != 1)
            {
                return false;
            }
            if (std::get<1>(gcd_tuple) < 0)
            {
                result = std::get<1>(gcd_tuple) + modulus;
                return true;
            }
            result = static_cast<std::uint64_t>(std::get<1>(gcd_tuple));
            return true;
        }

        std::vector<std::uint64_t> multiplicative_orders(std::vector<std::uint64_t> conjugate_classes, std::uint64_t modulus);

        std::vector<std::uint64_t> conjugate_classes(std::uint64_t modulus, std::uint64_t subgroup_generator);

        void babystep_giantstep(std::uint64_t modulus, std::vector<std::uint64_t> &baby_steps, std::vector<std::uint64_t> &giant_steps);

        std::pair<std::size_t, std::size_t> decompose_babystep_giantstep(std::uint64_t modulus, std::uint64_t input,
            const std::vector<std::uint64_t> &baby_steps, const std::vector<std::uint64_t> &giant_steps);
    }
}
#pragma once

#include "seal/util/globals.h"
#include "seal/smallmodulus.h"
#include "seal/util/defines.h"
#include <vector>
#include <stdexcept>
#include <map>

namespace seal
{
    /**
    Returns the default coefficients modulus for a given polynomial modulus degree.
    The polynomial modulus and the coefficient modulus obtained in this way should
    provide approdimately 128 bits of security against the best known attacks,
    assuming the standard deviation of the noise distribution is left to its default 
    value.

    @param[in] poly_modulus_degree The degree of the polynomial modulus
    @throws std::out_of_range if poly_modulus_degree is not 1024, 2048, 4096, 8192, 16384, or 32768
    */
    inline std::vector<SmallModulus> coeff_modulus_128(int poly_modulus_degree)
    {
        try
        {
            return util::global_variables::default_coeff_modulus_128.at(poly_modulus_degree);
        }
        catch (const std::exception &)
        {
            throw std::out_of_range("no default parameters found");
        }
        return {};
    }

    /**
    Returns the default coefficients modulus for a given polynomial modulus degree.
    The polynomial modulus and the coefficient modulus obtained in this way should
    provide approdimately 192 bits of security against the best known attacks,
    assuming the standard deviation of the noise distribution is left to its default
    value.

    @param[in] poly_modulus_degree The degree of the polynomial modulus
    @throws std::out_of_range if poly_modulus_degree is not 1024, 2048, 4096, 8192, 16384, or 32768
    */
    inline std::vector<SmallModulus> coeff_modulus_192(int poly_modulus_degree)
    {
        try
        {
            return util::global_variables::default_coeff_modulus_192.at(poly_modulus_degree);
        }
        catch (const std::exception &)
        {
            throw std::out_of_range("no default parameters found");
        }
        return {};
    }

    /**
    Returns a 60-bit coefficient modulus prime.

    @param[in] index The list index of the prime
    @throws std::out_of_range if index is not within [0, 64)
    */
    inline SmallModulus small_mods_60bit(int index)
    {
        try
        {
            return util::global_variables::small_mods_60bit.at(index);
        }
        catch (const std::exception &)
        {
            throw std::out_of_range("index out of range");
        }
        return 0;
    }

    /**
    Returns a 50-bit coefficient modulus prime.

    @param[in] index The list index of the prime
    @throws std::out_of_range if index is not within [0, 64)
    */
    inline SmallModulus small_mods_50bit(int index)
    {
        try
        {
            return util::global_variables::small_mods_50bit.at(index);
        }
        catch (const std::exception &)
        {
            throw std::out_of_range("index out of range");
        }
        return 0;
    }

    /**
    Returns a 40-bit coefficient modulus prime.

    @param[in] index The list index of the prime
    @throws std::out_of_range if index is not within [0, 64)
    */
    inline SmallModulus small_mods_40bit(int index)
    {
        try
        {
            return util::global_variables::small_mods_40bit.at(index);
        }
        catch (const std::exception &)
        {
            throw std::out_of_range("index out of range");
        }
        return 0;
    }

    /**
    Returns a 30-bit coefficient modulus prime.

    @param[in] index The list index of the prime
    @throws std::out_of_range if index is not within [0, 64)
    */
    inline SmallModulus small_mods_30bit(int index)
    {
        try
        {
            return util::global_variables::small_mods_30bit.at(index);
        }
        catch (const std::exception &)
        {
            throw std::out_of_range("index out of range");
        }
        return 0;
    }

    /**
    Returns the largest allowed decomposition bit count (60).
    */
    constexpr int dbc_max()
    {
        return SEAL_DBC_MAX;
    }

    /**
    Returns the smallest allowed decomposition bit count (1).
    */
    constexpr int dbc_min()
    {
        return SEAL_DBC_MIN;
    }
}
#pragma once

#include "seal/biguint.h"

namespace Microsoft
{
    namespace Research
    {
        namespace SEAL
        {
            ref class BigPoly;

            /**
            <summary>Represents an unsigned integer with a specified bit width.</summary>
            <remarks>
            <para>
            Represents an unsigned integer with a specified bit width. BigUInts are mutable and able to be resized. 
            The bit count for a BigUInt (which can be read with <see cref="BitCount"/>) is set initially by the 
            constructor and can be resized either explicitly with the <see cref="Resize()"/> function or implicitly
            with an assignment operation (e.g., one of the Set() functions). A rich set of unsigned integer operations 
            are provided by the BigUInt class, including comparison, traditional arithmetic (addition, subtraction, 
            multiplication, division), and modular arithmetic functions.
            </para>

            <para>
            The backing array for a BigUInt stores its unsigned integer value as a contiguous System::UInt64 array.
            Each System::UInt64 in the array sequentially represents 64-bits of the integer value, with the least
            significant quad-word storing the lower 64-bits and the order of the bits for each quad word dependent 
            on the architecture's System::UInt64 representation. The size of the array equals the bit count of the 
            BigUInt (which can be read with <see cref="BitCount"/>) rounded up to the next System::UInt64 boundary 
            (i.e., rounded up to the next 64-bit boundary). The <see cref="UInt64Count"/> property returns the number
            of System::UInt64 in the backing array. The <see cref="Pointer"/> property returns a pointer to the first
            System::UInt64 in the array. Additionally, the index property allows accessing the individual bytes of
            the integer value in a platform-independent way - for example, reading the third byte will always return 
            bits 16-24 of the BigUInt value regardless of the platform being little-endian or big-endian.
            </para>

            <para>
            Both the copy constructor and the Set function allocate more memory for the backing array when needed, 
            i.e. when the source BigUInt has a larger backing array than the destination. Conversely, when the destination
            backing array is already large enough, the data is only copied and the unnecessary higher order bits are set
            to zero. When new memory has to be allocated, only the significant bits of the source BigUInt are taken into 
            account. This is is important, because it avoids unnecessary zero bits to be included in the destination,
            which in some cases could accumulate and result in very large unnecessary allocations. However, sometimes 
            it is necessary to preserve the original size, even if some of the leading bits are zero. For this purpose 
            BigUInt contains functions <see cref="DuplicateFrom"/> and <see cref="DuplicateTo"/>, which create an exact 
            copy of the source BigUInt.
            </para>

            <para>
            An aliased BigUInt (which can be determined with <see cref="IsAlias"/>) is a special type of BigUInt that 
            does not manage its underlying System::UInt64 pointer used to store the value. An aliased BigUInt supports
            most of the same operations as a non-aliased BigUInt, including reading and writing the value, however 
            an aliased BigUInt does not internally allocate or deallocate its backing array and, therefore, does not 
            support resizing. Any attempt, either explicitly or implicitly, to resize the BigUInt will result in an 
            exception being thrown. Aliased BigUInt's are only created internally. Aliasing is useful in cases where 
            it is desirable to not have each BigUInt manage its own memory allocation and/or to prevent unnecessary 
            copying. For example, <see cref="BigPoly"/> uses aliased BigUInt's to return BigUInt representations of 
            its coefficients, where the aliasing allows read/writes to the BigUInt to refer directly to the coefficient's 
            corresponding region in the backing array of the <see cref="BigPoly"/>.
            </para>

            <para>
            In general, reading a BigUInt is thread-safe while mutating is not. Specifically, the backing array may be 
            freed whenever a resize occurs or the BigUInt is destroyed, which would invalidate the address returned by
            <see cref="Pointer"/>. When it is known that a resize will not occur, concurrent reading and mutating will 
            not inherently fail but it is possible for a read to see a partially updated value from a concurrent write.
            A non-aliased BigUInt allocates its backing array from the global (thread-safe) memory pool. Consequently, 
            creating or resizing a large number of BigUInt can result in a performance loss due to thread contention.
            </para>
            </remarks>
            */
            public ref class BigUInt : System::IEquatable<BigUInt^>, System::IComparable<BigUInt^>
            {
            public:
                /**
                <summary>Creates an empty BigUInt with zero bit width.</summary>
                <remarks>
                Creates an empty BigUInt with zero bit width. No memory is allocated by this constructor.
                </remarks>
                */
                BigUInt();

                /**
                <summary>Creates a zero-initialized BigUInt of the specified bit width.</summary>

                <param name="bitCount">The bit width</param>
                <exception cref="System::ArgumentException">if bitCount is negative</exception>
                */
                BigUInt(int bitCount);

                /**
                <summary>Creates a BigUInt of the specified bit width and initializes it with the unsigned hexadecimal 
                integer specified by the string.</summary>
                <remarks>
                Creates a BigUInt of the specified bit width and initializes it with the unsigned hexadecimal integer 
                specified by the string. The string must match the format returned by <see cref="ToString()"/> and must 
                consist of only the characters 0-9, A-F, or a-f, most-significant nibble first.
                </remarks>

                <param name="bitCount">The bit width</param>
                <param name="hexString">The hexadecimal integer string specifying the initial value</param>
                <exception cref="System::ArgumentNullException">If hexString is null</exception>
                <exception cref="System::ArgumentException">if bitCount is negative</exception>
                <exception cref="System::ArgumentException">If hexString does not adhere to the expected format</exception>
                */
                BigUInt(int bitCount, System::String ^hexString);

                /**
                <summary>Creates a BigUInt of the specified bit width and initializes it to the specified unsigned integer
                value.</summary>

                <param name="bitCount">The bit width</param>
                <param name="value">The initial value to set the BigUInt</param>
                <exception cref="System::ArgumentException">if bitCount is negative</exception>
                */
                BigUInt(int bitCount, System::UInt64 value);

                /**
                <summary>Creates a BigUInt initialized and minimally sized to fit the unsigned hexadecimal integer specified 
                by the string.</summary>

                <remarks>
                Creates a BigUInt initialized and minimally sized to fit the unsigned hexadecimal integer specified by 
                the string. The string matches the format returned by <see cref="ToString()"/> and must consist of only 
                the characters 0-9, A-F, or a-f, most-significant nibble first.
                </remarks>

                <param name="hexString">The hexadecimal integer string specifying the initial value</param>
                <exception cref="System::ArgumentNullException">If hexString is null</exception>
                <exception cref="System::ArgumentException">If hexString does not adhere to the expected format</exception>
                */
                BigUInt(System::String ^hexString);

                /**
                <summary>Creates a BigUInt initialized and minimally sized to fit the unsigned hexadecimal integer specified 
                by the <see cref="System::Numerics::BigInteger"/>.</summary>
                
                <param name="bigInteger">The initial value of the BigUInt</param>
                <exception cref="System::ArgumentNullException">If bigInteger is null</exception>
                */
                BigUInt(System::Numerics::BigInteger ^bigInteger);

                /**
                <summary>Creates a deep copy of a BigUInt.</summary>
                <remarks>
                Creates a deep copy of a BigUInt. The created BigUInt will have the same bit count and value as the original.
                </remarks>

                <param name="copy">The BigUInt to copy from</param>
                <exception cref="System::ArgumentNullException">If copy is null</exception>
                */
                BigUInt(BigUInt ^copy);

                /**
                <summary>Returns whether or not the BigUInt is an alias.</summary>

                <seealso cref="BigUInt">See BigUInt for a detailed description of aliased BigUInt.</seealso>
                */
                property bool IsAlias {
                    bool get();
                }

                /**
                <summary>Returns the bit count for the BigUInt.</summary>

                <seealso cref="GetSignificantBitCount()">See GetSignificantBitCount() to instead ignore leading zero bits.</seealso>
                */
                property int BitCount {
                    int get();
                }

                /**
                <summary>Returns the number of bytes in the backing array used to store the BigUInt value.</summary>

                <seealso cref="BigUInt">See BigUInt for a detailed description of the format of the backing array.</seealso>
                */
                property int ByteCount {
                    int get();
                }

                /**
                <summary>Returns the number of System::UInt64 in the backing array used to store the BigUInt value.</summary>

                <seealso cref="BigUInt">See BigUInt for a detailed description of the format of the backing array.</seealso>
                */
                property int UInt64Count {
                    int get();
                }

                /**
                <summary>Returns a pointer to the backing array storing the BigUInt value.</summary>
                <remarks>
                Returns a pointer to the backing array storing the BigUInt value. The pointer points to the beginning of 
                the backing array at the least-significant quad word. The pointer is valid only until the backing array is
                freed, which occurs when the BigUInt is resized or destroyed.
                </remarks>

                <seealso cref="UInt64Count">See UInt64Count to determine the number of System::UInt64 values in the
                backing array.</seealso>
                <seealso cref="BigUInt">See BigUInt for a detailed description of the format of the backing array.</seealso>
                */
                property System::UInt64 *Pointer {
                    System::UInt64 *get();
                }

                /**
                <summary>Returns whether or not the BigUInt has the value zero.</summary>
                */
                property bool IsZero {
                    bool get();
                }

                /**
                <summary>Gets/sets the byte at the corresponding byte index of the BigUInt's integer value.</summary>
                <remarks>
                Gets/sets the byte at the corresponding byte index of the BigUInt's integer value. The bytes of the BigUInt 
                are indexed least-significant byte first.
                </remarks>

                <param name="index">The index of the byte to get/set</param>
                <exception cref="System::ArgumentOutOfRangeException">If index is not within [0, <see cref="ByteCount"/>)</exception>
                <seealso cref="BigUInt">See BigUInt for a detailed description of the format of the backing array.</seealso>
                */
                property System::Byte default[int]{
                    System::Byte get(int index);
                    void set(int index, System::Byte value);
                }

                /**
                <summary>Returns the number of significant bits for the BigUInt.</summary>

                <seealso cref="BitCount">See BitCount to instead return the bit count regardless of leading zero bits.</seealso>
                */
                int GetSignificantBitCount();

                /**
                <summary>Overwrites the BigUInt with the value of the specified BigUInt, enlarging if needed to fit the assigned
                value.</summary>
                <remarks>
                Overwrites the BigUInt with the value of the specified BigUInt, enlarging if needed to fit the assigned value.
                Only significant bits are used to size the BigUInt.
                </remarks>

                <param name="assign">The BigUInt whose value should be assigned to the current BigUInt</param>
                <exception cref="System::ArgumentNullException">If assign is null</exception>
                <exception cref="System::InvalidOperationException">If BigUInt is an alias and the assigned BigUInt is too large to fit
                the current bit width</exception>
                */
                void Set(BigUInt ^assign);

                /**
                <summary>Overwrites the BigUInt with the unsigned hexadecimal value specified by the string, enlarging if needed to fit
                the assigned value.</summary>
                <remarks>
                Overwrites the BigUInt with the unsigned hexadecimal value specified by the string, enlarging if needed to fit the
                assigned value. The string must match the format returned by <see cref="ToString()"/> and must consist of only the
                characters 0-9, A-F, or a-f, most-significant nibble first.
                </remarks>

                <param name="assign">The hexadecimal integer string specifying the value to assign</param>
                <exception cref="System::ArgumentNullException">If assign is null</exception>
                <exception cref="System::ArgumentException">If assign does not adhere to the expected format</exception>
                <exception cref="System::InvalidOperationException">If BigUInt is an alias and the assigned value is too large to fit
                the current bit width</exception>
                */
                void Set(System::String ^assign);

                /**
                <summary>Overwrites the BigUInt with the specified integer value, enlarging if needed to fit the value.</summary>

                <param name="assign">The value to assign</param>
                <exception cref="System::InvalidOperationException">If BigUInt is an alias and the significant bit count of assign is
                too large to fit the current bit width</exception>
                */
                void Set(System::UInt64 assign);

                /**
                <summary>Sets the BigUInt value to zero.</summary>
                <remarks>
                Sets the BigUInt value to zero. This does not resize the BigUInt.
                </remarks>
                */
                void SetZero();

                /**
                <summary>Saves the BigUInt to an output stream.</summary>
                <remarks>
                Saves the BigUInt to an output stream. The full state of the BigUInt is serialized, including insignificant bits. The
                output is in binary format and not human-readable. The output stream must have the "binary" flag set.
                </remarks>

                <param name="stream">The stream to save the BigUInt to</param>
                <exception cref="System::ArgumentNullException">If stream is null</exception>
                <seealso cref="Load()">See Load() to load a saved BigUInt.</seealso>
                */
                void Save(System::IO::Stream ^stream);

                /**
                <summary>Loads a BigUInt from an input stream overwriting the current BigUInt and enlarging if needed to fit the loaded
                BigUInt.</summary>

                <param name="stream">The stream to load the BigUInt from</param>
                <exception cref="System::ArgumentNullException">If stream is null</exception>
                <exception cref="System::InvalidOperationException">If BigUInt is an alias and the loaded BigUInt is too large to fit
                with the current bit width</exception>
                <seealso cref="Save()">See Save() to save a BigUInt.</seealso>
                */
                void Load(System::IO::Stream ^stream);

                /**
                <summary>Resizes the BigUInt to the specified bit width, copying over the old value as much as will fit.</summary>

                <param name="bitCount">The bit width</param>
                <exception cref="System::ArgumentException">if bitCount is negative</exception>
                <exception cref="System::InvalidOperationException">If the BigUInt is an alias</exception>
                */
                void Resize(int bitCount);

                /**
                <summary>Returns whether or not a BigUInt is equal to a second BigUInt.</summary>
                <remarks>
                Returns whether or not a BigUInt is equal to a second BigUInt. The input operands are not modified.
                </remarks>

                <param name="compare">The value to compare against</param>
                <exception cref="System::ArgumentNullException">If compare is null</exception>
                */
                virtual bool Equals(BigUInt ^compare);

                /**
                <summary>Returns whether or not a BigUInt is equal to a second BigUInt.</summary>
                <remarks>
                Returns whether or not a BigUInt is equal to a second BigUInt. The input operands are not modified.
                </remarks>

                <param name="compare">The value to compare against</param>
                */
                bool Equals(System::Object ^compare) override;

                /**
                <summary>Returns the BigUInt value as a <see cref="System::Numerics::BigInteger"/>.</summary>
                */
                System::Numerics::BigInteger ^ToBigInteger();

                /**
                <summary>Returns the BigUInt value as a hexadecimal string.</summary>
                */
                System::String ^ToString() override;

                /**
                <summary>Returns the BigUInt value as a decimal string.</summary>
                */
                System::String ^ToDecimalString();

                /**
                <summary>Returns a hash-code based on the value of the BigUInt.</summary>
                */
                int GetHashCode() override;

                /**
                <summary>Destroys the BigUInt, including deallocating any internally allocated space.</summary>
                */
                ~BigUInt();

                /**
                <summary>Destroys the BigUInt, including deallocating any internally allocated space.</summary>
                */
                !BigUInt();

                /**
                <summary>Compares two BigUInts and returns -1, 0, or 1 if the BigUInt is less-than, equal-to, or greater-than the
                second operand respectively.</summary>

                <remarks>
                Compares two BigUInts and returns -1, 0, or 1 if the BigUInt is less-than, equal-to, or greater-than the second
                operand respectively. The input operands are not modified.
                </remarks>
                <param name="compare">The value to compare against</param>
                <exception cref="System::ArgumentNullException">If compare is null</exception>
                */
                virtual int CompareTo(BigUInt ^compare);

                /**
                <summary>Compares a BigUInt and an unsigned integer and returns -1, 0, or 1 if the BigUInt is less-than, equal-to, or
                greater-than the second operand respectively.</summary>

                <remarks>
                Compares a BigUInt and an unsigned integer and returns -1, 0, or 1 if the BigUInt is less-than, equal-to, or
                greater-than the second operand respectively. The input operands are not modified.
                </remarks>
                <param name="compare">The value to compare against</param>
                */
                virtual int CompareTo(System::UInt64 compare);

                /**
                <summary>Divides two BigUInts and returns the quotient and sets the remainder parameter to the remainder.</summary>

                <remarks>
                Divides two BigUInts and returns the quotient and sets the remainder parameter to the remainder. The bit count of the
                quotient is set to be the significant bit count of the BigUInt. The remainder is resized if and only if it is smaller
                than the bit count of the BigUInt.
                </remarks>
                <param name="operand2">The second operand to divide</param>
                <param name="remainder">The BigUInt to store the remainder</param>
                <exception cref="System::ArgumentNullException">If operand2 or remainder is null</exception>
                <exception cref="System::ArgumentException">If operand2 is zero</exception>
                <exception cref="System::InvalidOperationException">If the remainder is an alias and the operator attempts to enlarge
                the BigUInt to fit the result</exception>
                */
                BigUInt ^DivideRemainder(BigUInt ^operand2, BigUInt ^remainder);

                /**
                <summary>Divides a BigUInt and an unsigned integer and returns the quotient and sets the remainder parameter to the
                remainder.</summary>

                <remarks>
                Divides a BigUInt and an unsigned integer and returns the quotient and sets the remainder parameter to the remainder.
                The bit count of the quotient is set to be the significant bit count of the BigUInt. The remainder is resized if and
                only if it is smaller than the bit count of the BigUInt.
                </remarks>
                <param name="operand2">The second operand to divide</param>
                <param name="remainder">The BigUInt to store the remainder</param>
                <exception cref="System::ArgumentNullException">If remainder is null</exception>
                <exception cref="System::ArgumentException">If operand2 is zero</exception>
                <exception cref="System::InvalidOperationException">If the remainder is an alias which the function attempts to enlarge
                to fit the result</exception>
                */
                BigUInt ^DivideRemainder(System::UInt64 operand2, BigUInt ^remainder);

                /**
                <summary>Returns the inverse of a BigUInt with respect to the specified modulus.</summary>

                <remarks>
                Returns the inverse of a BigUInt with respect to the specified modulus. The original BigUInt is not modified. The bit
                count of the inverse is set to be the significant bit count of the modulus.
                </remarks>
                <param name="modulus">The modulus to calculate the inverse with respect to</param>
                <exception cref="System::ArgumentNullException">If modulus is null</exception>
                <exception cref="System::ArgumentException">If modulus is zero</exception>
                <exception cref="System::ArgumentException">If modulus is not greater than the BigUInt value</exception>
                <exception cref="System::ArgumentException">If the BigUInt value and modulus are not co-prime</exception>
                */
                BigUInt ^ModuloInvert(BigUInt ^modulus);

                /**
                <summary>Returns the inverse of a BigUInt with respect to the specified modulus.</summary>

                <remarks>
                Returns the inverse of a BigUInt with respect to the specified modulus. The original BigUInt is not modified. The bit
                count of the inverse is set to be the significant bit count of the modulus.
                </remarks>
                <param name="modulus">The modulus to calculate the inverse with respect to</param>
                <exception cref="System::ArgumentException">If modulus is zero</exception>
                <exception cref="System::ArgumentException">If modulus is not greater than the BigUInt value</exception>
                <exception cref="System::ArgumentException">If the BigUInt value and modulus are not co-prime</exception>
                */
                BigUInt ^ModuloInvert(System::UInt64 modulus);

                /**
                <summary>Attempts to calculate the inverse of a BigUInt with respect to the specified modulus, returning whether or not
                the inverse was successful and setting the inverse parameter to the inverse.</summary>

                <remarks>
                Attempts to calculate the inverse of a BigUInt with respect to the specified modulus, returning whether or not the
                inverse was successful and setting the inverse parameter to the inverse. The original BigUInt is not modified. The
                inverse parameter is resized if and only if its bit count is smaller than the significant bit count of the modulus.
                </remarks>
                <param name="modulus">The modulus to calculate the inverse with respect to</param>
                <param name="inverse">Stores the inverse if the inverse operation was successful</param>
                <exception cref="System::ArgumentNullException">If modulus or inverse is null</exception>
                <exception cref="System::ArgumentException">If modulus is zero</exception>
                <exception cref="System::ArgumentException">If modulus is not greater than the BigUInt value</exception>
                <exception cref="System::InvalidOperationException">If the inverse is an alias which the function attempts to enlarge
                to fit the result</exception>
                */
                bool TryModuloInvert(BigUInt ^modulus, BigUInt ^inverse);

                /**
                <summary>Attempts to calculate the inverse of a BigUInt with respect to the specified modulus, returning whether or not
                the inverse was successful and setting the inverse parameter to the inverse.</summary>

                <remarks>
                Attempts to calculate the inverse of a BigUInt with respect to the specified modulus, returning whether or not the
                inverse was successful and setting the inverse parameter to the inverse. The original BigUInt is not modified. The
                inverse parameter is resized if and only if its bit count is smaller than the significant bit count of the modulus.
                </remarks>
                <param name="modulus">The modulus to calculate the inverse with respect to</param>
                <param name="inverse">Stores the inverse if the inverse operation was successful</param>
                <exception cref="System::ArgumentNullException">If inverse is null</exception>
                <exception cref="System::ArgumentException">If modulus is zero</exception>
                <exception cref="System::ArgumentException">If modulus is not greater than the BigUInt value</exception>
                <exception cref="System::InvalidOperationException">If the inverse is an alias which the function attempts to enlarge
                to fit the result</exception>
                */
                bool TryModuloInvert(System::UInt64 modulus, BigUInt ^inverse);

                /**
                <summary>Returns a copy of the BigUInt value resized to the significant bit count.</summary>

                <param name="operand">The operand to copy</param>
                <exception cref="System::ArgumentNullException">If operand is null</exception>
                */
                static BigUInt ^operator +(BigUInt ^operand);

                /**
                <summary>Returns a negated copy of the BigUInt value.</summary>

                <remarks>
                Returns a negated copy of the BigUInt value. The bit count does not change.
                </remarks>
                <param name="operand">The operand to negate</param>
                <exception cref="System::ArgumentNullException">If operand is null</exception>
                */
                static BigUInt ^operator -(BigUInt ^operand);

                /**
                <summary>Returns an inverted copy of the BigUInt value.</summary>

                <remarks>
                Returns an inverted copy of the BigUInt value. The bit count does not change.
                </remarks>
                <param name="operand">The operand to invert</param>
                <exception cref="System::ArgumentNullException">If operand is null</exception>
                */
                static BigUInt ^operator ~(BigUInt ^operand);

                /**
                <summary>Increments the BigUInt and returns the incremented value.</summary>

                <remarks>
                Increments the BigUInt and returns the incremented value. The BigUInt will increment the bit count if needed to fit the
                carry.
                </remarks>
                <param name="operand">The operand to increment</param>
                <exception cref="System::ArgumentNullException">If operand is null</exception>
                <exception cref="System::InvalidOperationException">If BigUInt is an alias and a carry occurs requiring the BigUInt to
                be resized</exception>
                */
                static BigUInt ^operator ++(BigUInt ^operand);

                /**
                <summary>Decrements the BigUInt and returns the decremented value.</summary>

                <remarks>
                Decrements the BigUInt and returns the decremented value. The bit count does not change.
                </remarks>
                <param name="operand">The operand to decrement</param>
                <exception cref="System::ArgumentNullException">If operand is null</exception>
                */
                static BigUInt ^operator --(BigUInt ^operand);

                /**
                <summary>Adds two BigUInts and returns the sum.</summary>

                <remarks>
                Adds two BigUInts and returns the sum. The input operands are not modified. The bit count of the sum is set to be one
                greater than the significant bit count of the larger of the two input operands.
                </remarks>
                <param name="operand1">The first operand to add</param>
                <param name="operand2">The second operand to add</param>
                <exception cref="System::ArgumentNullException">If operand1 or operand2 is null</exception>
                */
                static BigUInt ^operator +(BigUInt ^operand1, BigUInt ^operand2);

                /**
                <summary>Adds a BigUInt and an unsigned integer and returns the sum.</summary>

                <remarks>
                Adds a BigUInt and an unsigned integer and returns the sum. The input operands are not modified. The bit count of the
                sum is set to be one greater than the significant bit count of the larger of the two operands.
                </remarks>
                <param name="operand1">The first operand to add</param>
                <param name="operand2">The second operand to add</param>
                <exception cref="System::ArgumentNullException">If operand1 is null</exception>
                */
                static BigUInt ^operator +(BigUInt ^operand1, System::UInt64 operand2);

                /**
                <summary>Subtracts two BigUInts and returns the difference.</summary>

                <remarks>
                Subtracts two BigUInts and returns the difference. The input operands are not modified. The bit count of the difference
                is set to be the significant bit count of the larger of the two input operands.
                </remarks>
                <param name="operand1">The first operand to subtract</param>
                <param name="operand2">The second operand to subtract</param>
                <exception cref="System::ArgumentNullException">If operand1 or operand2 is null</exception>
                */
                static BigUInt ^operator -(BigUInt ^operand1, BigUInt ^operand2);

                /**
                <summary>Subtracts a BigUInt and an unsigned integer and returns the difference.</summary>

                <remarks>
                Subtracts a BigUInt and an unsigned integer and returns the difference. The input operands are not modified. The bit
                count of the difference is set to be the significant bit count of the larger of the two operands.
                </remarks>
                <param name="operand1">The first operand to subtract</param>
                <param name="operand2">The second operand to subtract</param>
                <exception cref="System::ArgumentNullException">If operand1 is null</exception>
                */
                static BigUInt ^operator -(BigUInt ^operand1, System::UInt64 operand2);

                /**
                <summary>Multiplies two BigUInts and returns the product.</summary>

                <remarks>
                Multiplies two BigUInts and returns the product. The input operands are not modified. The bit count of the product is
                set to be the sum of the significant bit counts of the two input operands.
                </remarks>
                <param name="operand1">The first operand to multiply</param>
                <param name="operand2">The second operand to multiply</param>
                <exception cref="System::ArgumentNullException">If operand1 or operand2 is null</exception>
                */
                static BigUInt ^operator *(BigUInt ^operand1, BigUInt ^operand2);

                /**
                <summary>Multiplies a BigUInt and an unsigned integer and returns the product.</summary>

                <remarks>
                Multiplies a BigUInt and an unsigned integer and returns the product. The input operands are not modified. The bit
                count of the product is set to be the sum of the significant bit counts of the two input operands.
                </remarks>
                <param name="operand1">The first operand to multiply</param>
                <param name="operand2">The second operand to multiply</param>
                <exception cref="System::ArgumentNullException">If operand1 is null</exception>
                */
                static BigUInt ^operator *(BigUInt ^operand1, System::UInt64 operand2);

                /**
                <summary>Divides two BigUInts and returns the quotient.</summary>

                <remarks>
                Divides two BigUInts and returns the quotient. The input operands are not modified. The bit count of the quotient is
                set to be the significant bit count of the first input operand.
                </remarks>
                <param name="operand1">The first operand to divide</param>
                <param name="operand2">The second operand to divide</param>
                <exception cref="System::ArgumentNullException">If operand1 or operand2 is null</exception>
                <exception cref="System::ArgumentException">If operand2 is zero</exception>
                */
                static BigUInt ^operator /(BigUInt ^operand1, BigUInt ^operand2);

                /**
                <summary>Divides a BigUInt and an unsigned integer and returns the quotient.</summary>

                <remarks>
                Divides a BigUInt and an unsigned integer and returns the quotient. The input operands are not modified. The bit count
                of the quotient is set to be the significant bit count of the first input operand.
                </remarks>
                <param name="operand1">The first operand to divide</param>
                <param name="operand2">The second operand to divide</param>
                <exception cref="System::ArgumentNullException">If operand1 is null</exception>
                <exception cref="System::ArgumentException">If operand2 is zero</exception>
                */
                static BigUInt ^operator /(BigUInt ^operand1, System::UInt64 operand2);

                /**
                <summary>Divides two BigUInts and returns the remainder.</summary>

                <remarks>
                Divides two BigUInts and returns the remainder. The input operands are not modified. The bit count of the remainder is
                set to be the significant bit count of the first input operand.
                </remarks>
                <param name="operand1">The first operand to divide</param>
                <param name="operand2">The second operand to divide</param>
                <exception cref="System::ArgumentNullException">If operand1 or operand2 is null</exception>
                <exception cref="System::ArgumentException">If operand2 is zero</exception>
                */
                static BigUInt ^operator %(BigUInt ^operand1, BigUInt ^operand2);

                /**
                <summary>Divides a BigUInt and an unsigned integer and returns the remainder.</summary>

                <remarks>
                Divides a BigUInt and an unsigned integer and returns the remainder. The input operands are not modified. The bit count
                of the remainder is set to be the significant bit count of the first input operand.
                </remarks>
                <param name="operand1">The first operand to divide</param>
                <param name="operand2">The second operand to divide</param>
                <exception cref="System::ArgumentNullException">If operand1 is null</exception>
                <exception cref="System::ArgumentException">If operand2 is zero</exception>
                */
                static BigUInt ^operator %(BigUInt ^operand1, System::UInt64 operand2);

                /**
                <summary>Performs a bit-wise XOR operation between two BigUInts and returns the result.</summary>

                <remarks>
                Performs a bit-wise XOR operation between two BigUInts and returns the result. The input operands are not modified. The
                bit count of the result is set to the maximum of the two input operand bit counts.
                </remarks>
                <param name="operand1">The first operand to XOR</param>
                <param name="operand2">The second operand to XOR</param>
                <exception cref="System::ArgumentNullException">If operand1 or operand2 is null</exception>
                */
                static BigUInt ^operator ^(BigUInt ^operand1, BigUInt ^operand2);

                /**
                <summary>Performs a bit-wise XOR operation between a BigUInt and an unsigned integer and returns the result.</summary>

                <remarks>
                Performs a bit-wise XOR operation between a BigUInt and an unsigned integer and returns the result. The input operands
                are not modified. The bit count of the result is set to the maximum of the two input operand bit counts.
                </remarks>
                <param name="operand1">The first operand to XOR</param>
                <param name="operand2">The second operand to XOR</param>
                <exception cref="System::ArgumentNullException">If operand1 is null</exception>
                */
                static BigUInt ^operator ^(BigUInt ^operand1, System::UInt64 operand2);

                /**
                <summary>Performs a bit-wise AND operation between two BigUInts and returns the result.</summary>

                <remarks>
                Performs a bit-wise AND operation between two BigUInts and returns the result. The input operands are not modified. The
                bit count of the result is set to the maximum of the two input operand bit counts.
                </remarks>
                <param name="operand1">The first operand to AND</param>
                <param name="operand2">The second operand to AND</param>
                <exception cref="System::ArgumentNullException">If operand1 or operand2 is null</exception>
                */
                static BigUInt ^operator &(BigUInt ^operand1, BigUInt ^operand2);

                /**
                <summary>Performs a bit-wise AND operation between a BigUInt and an unsigned integer and returns the result.</summary>

                <remarks>
                Performs a bit-wise AND operation between a BigUInt and an unsigned integer and returns the result. The input operands
                are not modified. The bit count of the result is set to the maximum of the two input operand bit counts.
                </remarks>
                <param name="operand1">The first operand to AND</param>
                <param name="operand2">The second operand to AND</param>
                <exception cref="System::ArgumentNullException">If operand1 is null</exception>
                */
                static BigUInt ^operator &(BigUInt ^operand1, System::UInt64 operand2);

                /**
                <summary>Performs a bit-wise OR operation between two BigUInts and returns the result.</summary>

                <remarks>
                Performs a bit-wise OR operation between two BigUInts and returns the result. The input operands are not modified. The
                bit count of the result is set to the maximum of the two input operand bit counts.
                </remarks>
                <param name="operand1">The first operand to OR</param>
                <param name="operand2">The second operand to OR</param>
                <exception cref="System::ArgumentNullException">If operand1 or operand2 is null</exception>
                */
                static BigUInt ^operator |(BigUInt ^operand1, BigUInt ^operand2);

                /**
                <summary>Performs a bit-wise OR operation between a BigUInt and an unsigned integer and returns the result.</summary>

                <remarks>
                Performs a bit-wise OR operation between a BigUInt and an unsigned integer and returns the result. The input operands
                are not modified. The bit count of the result is set to the maximum of the two input operand bit counts.
                </remarks>
                <param name="operand1">The first operand to OR</param>
                <param name="operand2">The second operand to OR</param>
                <exception cref="System::ArgumentNullException">If operand1 is null</exception>
                */
                static BigUInt ^operator |(BigUInt ^operand1, System::UInt64 operand2);

                /**
                <summary>Returns a left-shifted copy of the BigUInt.</summary>

                <remarks>
                Returns a left-shifted copy of the BigUInt. The bit count of the returned value is the sum of the original significant
                bit count and the shift amount.
                </remarks>
                <param name="operand1">The operand to left-shift</param>
                <param name="shift">The number of bits to shift by</param>
                <exception cref="System::ArgumentNullException">If operand1 is null</exception>
                <exception cref="System::ArgumentException">If shift is negative</exception>
                */
                static BigUInt ^operator <<(BigUInt ^operand1, int shift);

                /**
                <summary>Returns a right-shifted copy of the BigUInt.</summary>

                <remarks>
                Returns a right-shifted copy of the BigUInt. The bit count of the returned value is the original significant bit count
                subtracted by the shift amount (clipped to zero if negative).
                </remarks>
                <param name="operand1">The operand to right-shift</param>
                <param name="shift">The number of bits to shift by</param>
                <exception cref="System::ArgumentNullException">If operand1 is null</exception>
                <exception cref="System::ArgumentException">If shift is negative</exception>
                */
                static BigUInt ^operator >>(BigUInt ^operand1, int shift);

                /**
                <summary>Returns the BigUInt value as a double.</summary>
                <remarks>
                Returns the BigUInt value as a double. Note that precision may be lost during the conversion.
                </remarks>
                <param name="value">The value to convert</param>
                <exception cref="System::ArgumentNullException">If value is null</exception>
                */
                static explicit operator double(BigUInt ^value);

                /**
                <summary>Returns the BigUInt value as a float.</summary>
                <remarks>
                Returns the BigUInt value as a float. Note that precision may be lost during the conversion.
                </remarks>
                <param name="value">The value to convert</param>
                <exception cref="System::ArgumentNullException">If value is null</exception>
                */
                static explicit operator float(BigUInt ^value);

                /**
                <summary>Returns the lower 64-bits of a BigUInt value.</summary>
                <remarks>
                Returns the lower 64-bits of a BigUInt value. Note that if the value is greater than 64-bits,
                the higher bits are dropped.
                </remarks>
                <param name="value">The value to convert</param>
                <exception cref="System::ArgumentNullException">If value is null</exception>
                */
                static explicit operator System::UInt64(BigUInt ^value);

                /**
                <summary>Returns the lower 64-bits of a BigUInt value as a signed-integer.</summary>
                <remarks>
                Returns the lower 64-bits of a BigUInt value as a signed-integer. Note that if the value is greater than
                64-bits, the result may be negative and the higher bits are dropped.
                </remarks>
                <param name="value">The value to convert</param>
                <exception cref="System::ArgumentNullException">If value is null</exception>
                */
                static explicit operator System::Int64(BigUInt ^value);

                /**
                <summary>Returns the lower 32-bits of a BigUInt value.</summary>
                <remarks>
                Returns the lower 32-bits of a BigUInt value. Note that if the value is greater than 32-bits,
                the higher bits are dropped.
                </remarks>
                <param name="value">The value to convert</param>
                <exception cref="System::ArgumentNullException">If value is null</exception>
                */
                static explicit operator System::UInt32(BigUInt ^value);

                /**
                <summary>Returns the lower 32-bits of a BigUInt value as a signed-integer.</summary>
                <remarks>
                Returns the lower 32-bits of a BigUInt value as a signed-integer. Note that if the value is greater than
                32-bits, the result may be negative and the higher bits are dropped.
                </remarks>
                <param name="value">The value to convert</param>
                <exception cref="System::ArgumentNullException">If value is null</exception>
                */
                static explicit operator System::Int32(BigUInt ^value);

                /**
                <summary>Creates a minimally sized BigUInt initialized to the specified unsigned integer value.</summary>

                <param name="value">The value to initialized the BigUInt to</param>
                <exception cref="System::ArgumentNullException">If value is null</exception>
                */
                static BigUInt ^Of(System::UInt64 value);

                /**
                <summary>Returns a reference to the underlying C++ BigUInt.</summary>
                */
                seal::BigUInt &GetUInt();

                /**
                <summary>Duplicates the current BigUInt.</summary>
                <remarks>
                Duplicates the current BigUInt. The bit count and the value of the given BigUInt are set to be exactly the same as in
                the current one.
                </remarks>
                <param name="destination">The BigUInt to overwrite with the duplicate</param>
                <exception cref="System::ArgumentNullException">if destination is null</exception>
                <exception cref="System::InvalidOperationException">if the destination BigUInt is an alias</exception>
                */
                void DuplicateTo(BigUInt ^destination);

                /**
                <summary>Duplicates a given BigUInt.</summary>
                <remarks>
                Duplicates a given BigUInt. The bit count and the value of the current BigUInt
                are set to be exactly the same as in the given one.
                </remarks>
                <param name="value">The BigUInt to duplicate</param>
                <exception cref="System::ArgumentNullException">if value is null</exception>
                <exception cref="System::InvalidOperationException">if the current BigUInt is an alias</exception>
                */
                void DuplicateFrom(BigUInt ^value);

                /**
                <summary>Creates a BigUInt initialized and minimally sized to fit the unsigned hexadecimal integer specified
                by the string.</summary>

                <remarks>
                Creates a BigUInt initialized and minimally sized to fit the unsigned hexadecimal integer specified by
                the string. The string matches the format returned by <see cref="ToString()"/> and must consist of only
                the characters 0-9, A-F, or a-f, most-significant nibble first.
                </remarks>

                <param name="hexString">The hexadecimal integer string specifying the initial value</param>
                <exception cref="System::ArgumentNullException">If hexString is null</exception>
                <exception cref="System::ArgumentException">If hexString does not adhere to the expected format</exception>
                */
                static operator BigUInt ^(System::String ^hexString);

            internal:
                /**
                <summary>Creates a deep copy of a C++ BigUInt.</summary>
                <remarks>
                Creates a deep copy of a BigUInt. The created BigUInt will have the same bit count and value as the original.
                </remarks>

                <param name="value">The BigUInt to copy from</param>
                */
                BigUInt(const seal::BigUInt &value);

                /**
                <summary>Initializes the BigUInt to use the specified C++ BigUInt.</summary>
                <remarks>
                Initializes the BigUInt to use the specified C++ BigUInt. This constructor does not copy the C++ BigUInt but actually
                uses the specified C++ BigUInt as the backing data. Upon destruction, the managed BigUInt will not destroy the C++
                BigUInt.
                </remarks>
                <param name="value">The BigUInt to use as the backing BigUInt</param>
                */
                BigUInt(seal::BigUInt *value);

            private:
                seal::BigUInt *biguint_;

                bool owned_;
            };
        }
    }
}

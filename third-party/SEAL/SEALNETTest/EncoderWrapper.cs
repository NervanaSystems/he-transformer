using System;
using Microsoft.Research.SEAL;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SEALNETTest
{
    [TestClass]
    public class EncoderWrapper
    {
        [TestMethod]
        public void BinaryEncodeDecodeBigUIntNET()
        {
            var modulus = new SmallModulus(0xFFFFFFFFFFFFFFF);
            var encoder = new BinaryEncoder(modulus, MemoryPoolHandle.New());

            var value = new BigUInt(64);
            value.Set("0");
            var poly = encoder.Encode(value);
            Assert.AreEqual(0, poly.SignificantCoeffCount());
            Assert.IsTrue(poly.IsZero);
            Assert.AreEqual(value, encoder.DecodeBigUInt(poly));

            value.Set("1");
            var poly1 = encoder.Encode(value);
            Assert.AreEqual(1, poly1.SignificantCoeffCount());
            Assert.AreEqual("1", poly1.ToString());
            Assert.AreEqual(value, encoder.DecodeBigUInt(poly1));

            value.Set("2");
            var poly2 = encoder.Encode(value);
            Assert.AreEqual(2, poly2.SignificantCoeffCount());
            Assert.AreEqual("1x^1", poly2.ToString());
            Assert.AreEqual(value, encoder.DecodeBigUInt(poly2));

            value.Set("3");
            var poly3 = encoder.Encode(value);
            Assert.AreEqual(2, poly3.SignificantCoeffCount());
            Assert.AreEqual("1x^1 + 1", poly3.ToString());
            Assert.AreEqual(value, encoder.DecodeBigUInt(poly3));

            value.Set("FFFFFFFFFFFFFFFF");
            var poly4 = encoder.Encode(value);
            Assert.AreEqual(64, poly4.SignificantCoeffCount());
            for (int i = 0; i < 64; ++i)
            {
                Assert.AreEqual("1", poly4[i].ToString());
            }
            Assert.AreEqual(value, encoder.DecodeBigUInt(poly4));

            value.Set("80F02");
            var poly5 = encoder.Encode(value);
            Assert.AreEqual(20, poly5.SignificantCoeffCount());
            for (int i = 0; i < 20; ++i)
            {
                if (i == 19 || (i >= 8 && i <= 11) || i == 1)
                {
                    Assert.AreEqual("1", poly5[i].ToString());
                }
                else
                {
                    Assert.IsTrue(poly5[i] == 0);
                }
            }
            Assert.AreEqual(value, encoder.DecodeBigUInt(poly5));

            var poly6 = new Plaintext(3);
            poly6[0] = 1;
            poly6[1] = 500;
            poly6[2] = 1023;
            value.Set(1 + 500 * 2 + 1023 * 4);
            Assert.AreEqual(value, encoder.DecodeBigUInt(poly6));

            modulus.Set(1024);
            var encoder2 = new BinaryEncoder(modulus, MemoryPoolHandle.New());
            var poly7 = new Plaintext(4);
            poly7[0] = 1023; // -1   (*1)
            poly7[1] = 512;  // -512 (*2)
            poly7[2] = 511;  // 511  (*4)
            poly7[3] = 1;    // 1    (*8)
            value.Set(-1 + -512 * 2 + 511 * 4 + 1 * 8);
            Assert.AreEqual(value, encoder2.DecodeBigUInt(poly7));
        }

        [TestMethod]
        public void BinaryEncodeDecodeUInt64NET()
        {
            var modulus = new SmallModulus(0xFFFFFFFFFFFFFFF);
            var encoder = new BinaryEncoder(modulus, MemoryPoolHandle.New());

            var poly = encoder.Encode(0UL);
            Assert.AreEqual(0, poly.SignificantCoeffCount());
            Assert.IsTrue(poly.IsZero);
            Assert.AreEqual(0UL, encoder.DecodeUInt64(poly));

            var poly1 = encoder.Encode(1UL);
            Assert.AreEqual(1, poly1.SignificantCoeffCount());
            Assert.AreEqual("1", poly1.ToString());
            Assert.AreEqual(1UL, encoder.DecodeUInt64(poly1));

            var poly2 = encoder.Encode(2UL);
            Assert.AreEqual(2, poly2.SignificantCoeffCount());
            Assert.AreEqual("1x^1", poly2.ToString());
            Assert.AreEqual(2UL, encoder.DecodeUInt64(poly2));

            var poly3 = encoder.Encode(3UL);
            Assert.AreEqual(2, poly3.SignificantCoeffCount());
            Assert.AreEqual("1x^1 + 1", poly3.ToString());
            Assert.AreEqual(3UL, encoder.DecodeUInt64(poly3));

            var poly4 = encoder.Encode(0xFFFFFFFFFFFFFFFFUL);
            Assert.AreEqual(64, poly4.SignificantCoeffCount());
            for (int i = 0; i < 64; ++i)
            {
                Assert.AreEqual("1", poly4[i].ToString());
            }
            Assert.AreEqual(0xFFFFFFFFFFFFFFFFUL, encoder.DecodeUInt64(poly4));

            var poly5 = encoder.Encode(0x80F02UL);
            Assert.AreEqual(20, poly5.SignificantCoeffCount());
            for (int i = 0; i < 20; ++i)
            {
                if (i == 19 || (i >= 8 && i <= 11) || i == 1)
                {
                    Assert.AreEqual("1", poly5[i].ToString());
                }
                else
                {
                    Assert.IsTrue(poly5[i] == 0);
                }
            }
            Assert.AreEqual(0x80F02UL, encoder.DecodeUInt64(poly5));

            var poly6 = new Plaintext(3);
            poly6[0] = 1;
            poly6[1] = 500;
            poly6[2] = 1023;
            Assert.AreEqual((1UL + 500 * 2 + 1023 * 4), encoder.DecodeUInt64(poly6));

            modulus.Set(1024);
            var encoder2 = new BinaryEncoder(modulus, MemoryPoolHandle.New());
            var poly7 = new Plaintext(4);
            poly7[0] = 1023; // -1   (*1)
            poly7[1] = 512;  // -512 (*2)
            poly7[2] = 511;  // 511  (*4)
            poly7[3] = 1;    // 1    (*8)
            Assert.AreEqual((ulong)(-1 + -512 * 2 + 511 * 4 + 1 * 8), encoder2.DecodeUInt64(poly7));
        }

        [TestMethod]
        public void BinaryEncodeDecodeUInt32NET()
        {
            var modulus = new SmallModulus(0xFFFFFFFFFFFFFFF);
            var encoder = new BinaryEncoder(modulus, MemoryPoolHandle.New());

            var poly = encoder.Encode(0U);
            Assert.AreEqual(0, poly.SignificantCoeffCount());
            Assert.IsTrue(poly.IsZero);
            Assert.AreEqual(0U, encoder.DecodeUInt32(poly));

            var poly1 = encoder.Encode(1U);
            Assert.AreEqual(1, poly1.SignificantCoeffCount());
            Assert.AreEqual("1", poly1.ToString());
            Assert.AreEqual(1U, encoder.DecodeUInt32(poly1));

            var poly2 = encoder.Encode(2U);
            Assert.AreEqual(2, poly2.SignificantCoeffCount());
            Assert.AreEqual("1x^1", poly2.ToString());
            Assert.AreEqual(2U, encoder.DecodeUInt32(poly2));

            var poly3 = encoder.Encode(3U);
            Assert.AreEqual(2, poly3.SignificantCoeffCount());
            Assert.AreEqual("1x^1 + 1", poly3.ToString());
            Assert.AreEqual(3U, encoder.DecodeUInt32(poly3));

            var poly4 = encoder.Encode(0xFFFFFFFFU);
            Assert.AreEqual(32, poly4.SignificantCoeffCount());
            for (int i = 0; i < 32; ++i)
            {
                Assert.IsTrue("1" == poly4[i].ToString());
            }
            Assert.AreEqual(0xFFFFFFFFU, encoder.DecodeUInt32(poly4));

            var poly5 = encoder.Encode(0x80F02U);
            Assert.AreEqual(20, poly5.SignificantCoeffCount());
            for (int i = 0; i < 20; ++i)
            {
                if (i == 19 || (i >= 8 && i <= 11) || i == 1)
                {
                    Assert.AreEqual("1", poly5[i].ToString());
                }
                else
                {
                    Assert.IsTrue(poly5[i] == 0);
                }
            }
            Assert.AreEqual(0x80F02U, encoder.DecodeUInt32(poly5));

            var poly6 = new Plaintext(3);
            poly6[0] = 1;
            poly6[1] = 500;
            poly6[2] = 1023;
            Assert.AreEqual(1U + 500 * 2 + 1023 * 4, encoder.DecodeUInt32(poly6));

            modulus.Set(1024);
            var encoder2 = new BinaryEncoder(modulus, MemoryPoolHandle.New());
            var poly7 = new Plaintext(4);
            poly7[0] = 1023; // -1   (*1)
            poly7[1] = 512;  // -512 (*2)
            poly7[2] = 511;  // 511  (*4)
            poly7[3] = 1;    // 1    (*8)
            Assert.AreEqual((uint)(-1 + -512 * 2 + 511 * 4 + 1 * 8), encoder2.DecodeUInt32(poly7));
        }

        [TestMethod]
        public void BinaryEncodeDecodeInt64NET()
        {
            var modulus = new SmallModulus(0xFFFFFFFFFFFFFFF);
            var encoder = new BinaryEncoder(modulus, MemoryPoolHandle.New());

            var poly = encoder.Encode(0L);
            Assert.AreEqual(0, poly.SignificantCoeffCount());
            Assert.IsTrue(poly.IsZero);
            Assert.AreEqual(0L, encoder.DecodeInt64(poly));

            var poly1 = encoder.Encode(1L);
            Assert.AreEqual(1, poly1.SignificantCoeffCount());
            Assert.AreEqual("1", poly1.ToString());
            Assert.AreEqual(1L, encoder.DecodeInt64(poly1));

            var poly2 = encoder.Encode(2L);
            Assert.AreEqual(2, poly2.SignificantCoeffCount());
            Assert.AreEqual("1x^1", poly2.ToString());
            Assert.AreEqual(2L, encoder.DecodeInt64(poly2));

            var poly3 = encoder.Encode(3L);
            Assert.AreEqual(2, poly3.SignificantCoeffCount());
            Assert.AreEqual("1x^1 + 1", poly3.ToString());
            Assert.AreEqual(3L, encoder.DecodeInt64(poly3));

            var poly4 = encoder.Encode(-1L);
            Assert.AreEqual(1, poly4.SignificantCoeffCount());
            Assert.AreEqual("FFFFFFFFFFFFFFE", poly4.ToString());
            Assert.AreEqual(-1L, encoder.DecodeInt64(poly4));

            var poly5 = encoder.Encode(-2L);
            Assert.AreEqual(2, poly5.SignificantCoeffCount());
            Assert.AreEqual("FFFFFFFFFFFFFFEx^1", poly5.ToString());
            Assert.AreEqual(-2L, encoder.DecodeInt64(poly5));

            var poly6 = encoder.Encode(-3L);
            Assert.AreEqual(2, poly6.SignificantCoeffCount());
            Assert.AreEqual("FFFFFFFFFFFFFFEx^1 + FFFFFFFFFFFFFFE", poly6.ToString());
            Assert.AreEqual(-3L, encoder.DecodeInt64(poly6));

            var poly7 = encoder.Encode(0x7FFFFFFFFFFFFFFFL);
            Assert.AreEqual(63, poly7.SignificantCoeffCount());
            for (int i = 0; i < 63; ++i)
            {
                Assert.AreEqual(1UL, poly7[i]);
            }
            Assert.AreEqual(0x7FFFFFFFFFFFFFFFL, encoder.DecodeInt64(poly7));

            var poly8 = encoder.Encode(unchecked((long)0x8000000000000000));
            Assert.AreEqual(64, poly8.SignificantCoeffCount());
            Assert.AreEqual(0xFFFFFFFFFFFFFFEUL, poly8[63]);
            for (int i = 0; i < 63; ++i)
            {
                Assert.IsTrue(poly8[i] == 0);
            }
            Assert.AreEqual(unchecked((long)0x8000000000000000), encoder.DecodeInt64(poly8));

            var poly9 = encoder.Encode(0x80F02L);
            Assert.AreEqual(20, poly9.SignificantCoeffCount());
            for (int i = 0; i < 20; ++i)
            {
                if (i == 19 || (i >= 8 && i <= 11) || i == 1)
                {
                    Assert.AreEqual(1UL, poly9[i]);
                }
                else
                {
                    Assert.IsTrue(poly9[i] == 0);
                }
            }
            Assert.AreEqual(0x80F02L, encoder.DecodeInt64(poly9));

            var poly10 = encoder.Encode(-1073L);
            Assert.AreEqual(11, poly10.SignificantCoeffCount());
            Assert.AreEqual(0xFFFFFFFFFFFFFFEUL, poly10[10]);
            Assert.IsTrue(poly10[9] == 0);
            Assert.IsTrue(poly10[8] == 0);
            Assert.IsTrue(poly10[7] == 0);
            Assert.IsTrue(poly10[6] == 0);
            Assert.AreEqual(0xFFFFFFFFFFFFFFEUL, poly10[5]);
            Assert.AreEqual(0xFFFFFFFFFFFFFFEUL, poly10[4]);
            Assert.IsTrue(poly10[3] == 0);
            Assert.IsTrue(poly10[2] == 0);
            Assert.IsTrue(poly10[1] == 0);
            Assert.AreEqual(0xFFFFFFFFFFFFFFEUL, poly10[0]);
            Assert.AreEqual(-1073L, encoder.DecodeInt64(poly10));

            modulus.Set(0xFFFF);
            var encoder2 = new BinaryEncoder(modulus, MemoryPoolHandle.New());
            var poly11 = new Plaintext(6);
            poly11[0] = 1;
            poly11[1] = 0xFFFE; // -1
            poly11[2] = 0xFFFD; // -2
            poly11[3] = 0x8000; // -32767
            poly11[4] = 0x7FFF; // 32767
            poly11[5] = 0x7FFE; // 32766
            Assert.AreEqual(1L + -1 * 2 +  -2 * 4 + -32767 * 8 + 32767 * 16 + 32766 * 32, encoder2.DecodeInt64(poly11));
        }

        [TestMethod]
        public void BinaryEncodeDecodeInt32NET()
        {
            var modulus = new SmallModulus(0xFFFFFFFFFFFFFFF);
            var encoder = new BinaryEncoder(modulus, MemoryPoolHandle.New());

            var poly = encoder.Encode(0);
            Assert.AreEqual(0, poly.SignificantCoeffCount());
            Assert.IsTrue(poly.IsZero);
            Assert.AreEqual(0, encoder.DecodeInt32(poly));

            var poly1 = encoder.Encode(1);
            Assert.AreEqual(1, poly1.SignificantCoeffCount());
            Assert.AreEqual("1", poly1.ToString());
            Assert.AreEqual(1, encoder.DecodeInt32(poly1));

            var poly2 = encoder.Encode(2);
            Assert.AreEqual(2, poly2.SignificantCoeffCount());
            Assert.AreEqual("1x^1", poly2.ToString());
            Assert.AreEqual(2, encoder.DecodeInt32(poly2));

            var poly3 = encoder.Encode(3);
            Assert.AreEqual(2, poly3.SignificantCoeffCount());
            Assert.AreEqual("1x^1 + 1", poly3.ToString());
            Assert.AreEqual(3, encoder.DecodeInt32(poly3));

            var poly4 = encoder.Encode(-1);
            Assert.AreEqual(1, poly4.SignificantCoeffCount());
            Assert.AreEqual("FFFFFFFFFFFFFFE", poly4.ToString());
            Assert.AreEqual(-1, encoder.DecodeInt32(poly4));

            var poly5 = encoder.Encode(-2);
            Assert.AreEqual(2, poly5.SignificantCoeffCount());
            Assert.AreEqual("FFFFFFFFFFFFFFEx^1", poly5.ToString());
            Assert.AreEqual(-2, encoder.DecodeInt32(poly5));

            var poly6 = encoder.Encode(-3);
            Assert.AreEqual(2, poly6.SignificantCoeffCount());
            Assert.AreEqual("FFFFFFFFFFFFFFEx^1 + FFFFFFFFFFFFFFE", poly6.ToString());
            Assert.AreEqual(-3, encoder.DecodeInt32(poly6));

            var poly7 = encoder.Encode(0x7FFFFFFF);
            Assert.AreEqual(31, poly7.SignificantCoeffCount());
            for (int i = 0; i < 31; ++i)
            {
                Assert.AreEqual("1", poly7[i].ToString());
            }
            Assert.AreEqual(0x7FFFFFFF, encoder.DecodeInt32(poly7));

            var poly8 = encoder.Encode(unchecked((int)0x80000000));
            Assert.AreEqual(32, poly8.SignificantCoeffCount());
            Assert.AreEqual(0xFFFFFFFFFFFFFFEUL, poly8[31]);
            for (int i = 0; i < 31; ++i)
            {
                Assert.IsTrue(poly8[i] == 0);
            }
            Assert.AreEqual(unchecked((int)0x80000000), encoder.DecodeInt32(poly8));

            var poly9 = encoder.Encode(0x80F02);
            Assert.AreEqual(20, poly9.SignificantCoeffCount());
            for (int i = 0; i < 20; ++i)
            {
                if (i == 19 || (i >= 8 && i <= 11) || i == 1)
                {
                    Assert.AreEqual("1", poly9[i].ToString());
                }
                else
                {
                    Assert.IsTrue(poly9[i] == 0);
                }
            }
            Assert.AreEqual(0x80F02, encoder.DecodeInt32(poly9));

            var poly10 = encoder.Encode(-1073);
            Assert.AreEqual(11, poly10.SignificantCoeffCount());
            Assert.AreEqual(0xFFFFFFFFFFFFFFEUL, poly10[10]);
            Assert.IsTrue(poly10[9] == 0);
            Assert.IsTrue(poly10[8] == 0);
            Assert.IsTrue(poly10[7] == 0);
            Assert.IsTrue(poly10[6] == 0);
            Assert.AreEqual(0xFFFFFFFFFFFFFFEUL, poly10[5]);
            Assert.AreEqual(0xFFFFFFFFFFFFFFEUL, poly10[4]);
            Assert.IsTrue(poly10[3] == 0);
            Assert.IsTrue(poly10[2] == 0);
            Assert.IsTrue(poly10[1] == 0);
            Assert.AreEqual(0xFFFFFFFFFFFFFFEUL, poly10[0]);
            Assert.AreEqual(-1073, encoder.DecodeInt32(poly10));

            modulus.Set(0xFFFF);
            var encoder2 = new BinaryEncoder(modulus, MemoryPoolHandle.New());
            var poly11 = new Plaintext(6);
            poly11[0] = 1;
            poly11[1] = 0xFFFE; // -1
            poly11[2] = 0xFFFD; // -2
            poly11[3] = 0x8000; // -32767
            poly11[4] = 0x7FFF; // 32767
            poly11[5] = 0x7FFE; // 32766
            Assert.AreEqual(1 + -1 * 2 + -2 * 4 + -32767 * 8 + 32767 * 16 + 32766 * 32, encoder2.DecodeInt32(poly11));
        }

        [TestMethod]
        public void BalancedEncodeDecodeBigUIntNET()
        {
            var modulus = new SmallModulus(0x10000);
            var encoder = new BalancedEncoder(modulus, 3, MemoryPoolHandle.New());

            var value = new BigUInt(64);
            value.Set("0");
            var poly = encoder.Encode(value);
            Assert.AreEqual(0, poly.SignificantCoeffCount());
            Assert.IsTrue(poly.IsZero);
            Assert.AreEqual(value, encoder.DecodeBigUInt(poly));

            value.Set("1");
            var poly1 = encoder.Encode(value);
            Assert.AreEqual(1, poly1.SignificantCoeffCount());
            Assert.AreEqual("1", poly1.ToString());
            Assert.AreEqual(value, encoder.DecodeBigUInt(poly1));

            value.Set("2");
            var poly2 = encoder.Encode(value);
            Assert.AreEqual(2, poly2.SignificantCoeffCount());
            Assert.AreEqual("1x^1 + FFFF", poly2.ToString());
            Assert.AreEqual(value, encoder.DecodeBigUInt(poly2));

            value.Set("3");
            var poly3 = encoder.Encode(value);
            Assert.AreEqual(2, poly3.SignificantCoeffCount());
            Assert.AreEqual("1x^1", poly3.ToString());
            Assert.AreEqual(value, encoder.DecodeBigUInt(poly3));

            value.Set("2671");
            var poly4 = encoder.Encode(value);
            Assert.AreEqual(9, poly4.SignificantCoeffCount());
            for (int i = 0; i < 9; ++i)
            {
                Assert.AreEqual("1", poly4[i].ToString());
            }
            Assert.AreEqual(value, encoder.DecodeBigUInt(poly4));

            value.Set("D4EB");
            var poly5 = encoder.Encode(value);
            Assert.AreEqual(11, poly5.SignificantCoeffCount());
            for (int i = 0; i < 11; ++i)
            {
                if (i%3 == 1)
                {
                    Assert.AreEqual(0x1UL, poly5[i]);
                }
                else if (i%3 == 0)
                {
                    Assert.IsTrue(poly5[i] == 0);
                }
                else
                {
                    Assert.AreEqual(0xFFFFUL, poly5[i]);
                }
            }
            Assert.AreEqual(value, encoder.DecodeBigUInt(poly5));

            var poly6 = new Plaintext(3);
            poly6[0] = 1;
            poly6[1] = 500;
            poly6[2] = 1023;
            value.Set(1 + 500 * 3 + 1023 * 9);
            Assert.AreEqual(value, encoder.DecodeBigUInt(poly6));

            var encoder2 = new BalancedEncoder(modulus, 7, MemoryPoolHandle.New());
            var poly7 = new Plaintext(4);
            poly7[0] = 123; // 123   (*1)
            poly7[1] = 0xFFFF;  // -1 (*7)
            poly7[2] = 511;  // 511  (*49)
            poly7[3] = 1;    // 1    (*343)
            value.Set(123 + -1 * 7 + 511 * 49 + 1 * 343);
            Assert.AreEqual(value, encoder2.DecodeBigUInt(poly7));

            var encoder3 = new BalancedEncoder(modulus, 6, MemoryPoolHandle.New());
            var poly8 = new Plaintext(4);
            poly8[0] = 5;
            poly8[1] = 4;
            poly8[2] = 3;
            poly8[3] = 2;
            value.Set(5 + 4 * 6 + 3 * 36 + 2 * 216);
            Assert.AreEqual(value, encoder3.DecodeBigUInt(poly8));

            var encoder4 = new BalancedEncoder(modulus, 10, MemoryPoolHandle.New());
            var poly9 = new Plaintext(4);
            poly9[0] = 1;
            poly9[1] = 2;
            poly9[2] = 3;
            poly9[3] = 4;
            value.Set(4321);
            Assert.AreEqual(value, encoder4.DecodeBigUInt(poly9));

            value.Set("4D2");
            var poly10 = encoder2.Encode(value);
            Assert.AreEqual(5, poly10.SignificantCoeffCount());
            Assert.IsTrue(value.Equals(encoder2.DecodeBigUInt(poly10)));

            value.Set("4D2");
            var poly11 = encoder3.Encode(value);
            Assert.AreEqual(5, poly11.SignificantCoeffCount());
            Assert.IsTrue(value.Equals(encoder3.DecodeBigUInt(poly11)));

            value.Set("4D2");
            var poly12 = encoder4.Encode(value);
            Assert.AreEqual(4, poly12.SignificantCoeffCount());
            Assert.IsTrue(value.Equals(encoder4.DecodeBigUInt(poly12)));
        }

        [TestMethod]
        public void BalancedEncodeDecodeUInt64NET()
        {
            var modulus = new SmallModulus(0x10000);
            var encoder = new BalancedEncoder(modulus, 3, MemoryPoolHandle.New());

            var poly = encoder.Encode(0UL);
            Assert.AreEqual(0, poly.SignificantCoeffCount());
            Assert.IsTrue(poly.IsZero);
            Assert.AreEqual(0UL, encoder.DecodeUInt64(poly));

            var poly1 = encoder.Encode(1UL);
            Assert.AreEqual(1, poly1.SignificantCoeffCount());
            Assert.AreEqual("1", poly1.ToString());
            Assert.AreEqual(1UL, encoder.DecodeUInt64(poly1));

            var poly2 = encoder.Encode(2UL);
            Assert.AreEqual(2, poly2.SignificantCoeffCount());
            Assert.AreEqual("1x^1 + FFFF", poly2.ToString());
            Assert.AreEqual(2UL, encoder.DecodeUInt64(poly2));

            var poly3 = encoder.Encode(3UL);
            Assert.AreEqual(2, poly3.SignificantCoeffCount());
            Assert.AreEqual("1x^1", poly3.ToString());
            Assert.AreEqual(3UL, encoder.DecodeUInt64(poly3));

            var poly4 = encoder.Encode(0x2671UL);
            Assert.AreEqual(9, poly4.SignificantCoeffCount());
            for (int i = 0; i < 9; ++i)
            {
                Assert.AreEqual(1UL, poly4[i]);
            }
            Assert.AreEqual(0x2671UL, encoder.DecodeUInt64(poly4));

            var poly5 = encoder.Encode(0xD4EBUL);
            Assert.AreEqual(11, poly5.SignificantCoeffCount());
            for (int i = 0; i < 11; ++i)
            {
                if (i % 3 == 1)
                {
                    Assert.AreEqual(1UL, poly5[i]);
                }
                else if (i % 3 == 0)
                {
                    Assert.IsTrue(poly5[i] == 0);
                }
                else
                {
                    Assert.AreEqual(0xFFFFUL, poly5[i]);
                }
            }
            Assert.AreEqual(0xD4EBUL, encoder.DecodeUInt64(poly5));

            var poly6 = new Plaintext(3);
            poly6[0] = 1;
            poly6[1] = 500;
            poly6[2] = 1023;
            Assert.AreEqual((1UL + 500 * 3 + 1023 * 9), encoder.DecodeUInt64(poly6));

            var encoder2 = new BalancedEncoder(modulus, 7, MemoryPoolHandle.New());
            var poly7 = new Plaintext(4);
            poly7[0] = 123; // 123   (*1)
            poly7[1] = 0xFFFF;  // -1 (*7)
            poly7[2] = 511;  // 511  (*49)
            poly7[3] = 1;    // 1    (*343)
            Assert.AreEqual((UInt64)(123 + -1 * 7 + 511 * 49 + 1 * 343), encoder2.DecodeUInt64(poly7));

            var encoder3 = new BalancedEncoder(modulus, 6, MemoryPoolHandle.New());
            var poly8 = new Plaintext(4);
            poly8[0] = 5;
            poly8[1] = 4;
            poly8[2] = 3;
            poly8[3] = 2;
            UInt64 value = 5 + 4 * 6 + 3 * 36 + 2 * 216;
            Assert.AreEqual(value, encoder3.DecodeUInt64(poly8));

            var encoder4 = new BalancedEncoder(modulus, 10, MemoryPoolHandle.New());
            var poly9 = new Plaintext(4);
            poly9[0] = 1;
            poly9[1] = 2;
            poly9[2] = 3;
            poly9[3] = 4;
            value = 4321;
            Assert.AreEqual(value, encoder4.DecodeUInt64(poly9));

            value = 1234;
            var poly10 = encoder2.Encode(value);
            Assert.AreEqual(5, poly10.SignificantCoeffCount());
            Assert.IsTrue(value.Equals(encoder2.DecodeUInt64(poly10)));

            value = 1234;
            var poly11 = encoder3.Encode(value);
            Assert.AreEqual(5, poly11.SignificantCoeffCount());
            Assert.IsTrue(value.Equals(encoder3.DecodeUInt64(poly11)));

            value = 1234;
            var poly12 = encoder4.Encode(value);
            Assert.AreEqual(4, poly12.SignificantCoeffCount());
            Assert.IsTrue(value.Equals(encoder4.DecodeUInt64(poly12)));
        }

        [TestMethod]
        public void BalancedEncodeDecodeUInt32NET()
        {
            var modulus = new SmallModulus(0x10000);
            var encoder = new BalancedEncoder(modulus, 3, MemoryPoolHandle.New());

            var poly = encoder.Encode(0U);
            Assert.AreEqual(0, poly.SignificantCoeffCount());
            Assert.IsTrue(poly.IsZero);
            Assert.AreEqual(0U, encoder.DecodeUInt32(poly));

            var poly1 = encoder.Encode(1U);
            Assert.AreEqual(1, poly1.SignificantCoeffCount());
            Assert.AreEqual("1", poly1.ToString());
            Assert.AreEqual(1U, encoder.DecodeUInt32(poly1));

            var poly2 = encoder.Encode(2U);
            Assert.AreEqual(2, poly2.SignificantCoeffCount());
            Assert.AreEqual("1x^1 + FFFF", poly2.ToString());
            Assert.AreEqual(2U, encoder.DecodeUInt32(poly2));

            var poly3 = encoder.Encode(3U);
            Assert.AreEqual(2, poly3.SignificantCoeffCount());
            Assert.AreEqual("1x^1", poly3.ToString());
            Assert.AreEqual(3U, encoder.DecodeUInt32(poly3));

            var poly4 = encoder.Encode(0x2671U);
            Assert.AreEqual(9, poly4.SignificantCoeffCount());
            for (int i = 0; i < 9; ++i)
            {
                Assert.AreEqual(1UL, poly4[i]);
            }
            Assert.AreEqual(0x2671U, encoder.DecodeUInt32(poly4));

            var poly5 = encoder.Encode(0xD4EBU);
            Assert.AreEqual(11, poly5.SignificantCoeffCount());
            for (int i = 0; i < 11; ++i)
            {
                if (i % 3 == 1)
                {
                    Assert.AreEqual(1UL, poly5[i]);
                }
                else if (i % 3 == 0)
                {
                    Assert.IsTrue(poly5[i] == 0);
                }
                else
                {
                    Assert.AreEqual(0xFFFFUL, poly5[i]);
                }
            }
            Assert.AreEqual(0xD4EBU, encoder.DecodeUInt32(poly5));

            var poly6 = new Plaintext(3);
            poly6[0] = 1;
            poly6[1] = 500;
            poly6[2] = 1023;
            Assert.AreEqual(1U + 500 * 3 + 1023 * 9, encoder.DecodeUInt32(poly6));

            var encoder2 = new BalancedEncoder(modulus, 7, MemoryPoolHandle.New());
            var poly7 = new Plaintext(4);
            poly7[0] = 123; // 123   (*1)
            poly7[1] = 0xFFFF;  // -1 (*7)
            poly7[2] = 511;  // 511  (*49)
            poly7[3] = 1;    // 1    (*343)
            Assert.AreEqual((UInt32)(123 + -1 * 7 + 511 * 49 + 1 * 343), encoder2.DecodeUInt32(poly7));

            var encoder3 = new BalancedEncoder(modulus, 6, MemoryPoolHandle.New());
            var poly8 = new Plaintext(4);
            poly8[0] = 5;
            poly8[1] = 4;
            poly8[2] = 3;
            poly8[3] = 2;
            UInt64 value = 5 + 4 * 6 + 3 * 36 + 2 * 216;
            Assert.AreEqual(value, encoder3.DecodeUInt32(poly8));

            var encoder4 = new BalancedEncoder(modulus, 10, MemoryPoolHandle.New());
            var poly9 = new Plaintext(4);
            poly9[0] = 1;
            poly9[1] = 2;
            poly9[2] = 3;
            poly9[3] = 4;
            value = 4321;
            Assert.AreEqual(value, encoder4.DecodeUInt32(poly9));

            value = 1234;
            var poly10 = encoder2.Encode(value);
            Assert.AreEqual(5, poly10.SignificantCoeffCount());
            Assert.IsTrue(value.Equals(encoder2.DecodeUInt32(poly10)));

            value = 1234;
            var poly11 = encoder3.Encode(value);
            Assert.AreEqual(5, poly11.SignificantCoeffCount());
            Assert.IsTrue(value.Equals(encoder3.DecodeUInt32(poly11)));

            value = 1234;
            var poly12 = encoder4.Encode(value);
            Assert.AreEqual(4, poly12.SignificantCoeffCount());
            Assert.IsTrue(value.Equals(encoder4.DecodeUInt32(poly12)));
        }

        [TestMethod]
        public void BalancedEncodeDecodeInt64NET()
        {
            var modulus = new SmallModulus(0x10000);
            var encoder = new BalancedEncoder(modulus, 3, MemoryPoolHandle.New());

            var poly = encoder.Encode(0L);
            Assert.AreEqual(0, poly.SignificantCoeffCount());
            Assert.IsTrue(poly.IsZero);
            Assert.AreEqual(0L, encoder.DecodeInt64(poly));

            var poly1 = encoder.Encode(1L);
            Assert.AreEqual(1, poly1.SignificantCoeffCount());
            Assert.AreEqual("1", poly1.ToString());
            Assert.AreEqual(1L, encoder.DecodeInt64(poly1));

            var poly2 = encoder.Encode(2L);
            Assert.AreEqual(2, poly2.SignificantCoeffCount());
            Assert.AreEqual("1x^1 + FFFF", poly2.ToString());
            Assert.AreEqual(2L, encoder.DecodeInt64(poly2));

            var poly3 = encoder.Encode(3L);
            Assert.AreEqual(2, poly3.SignificantCoeffCount());
            Assert.AreEqual("1x^1", poly3.ToString());
            Assert.AreEqual(3L, encoder.DecodeInt64(poly3));

            var poly4 = encoder.Encode(-1L);
            Assert.AreEqual(1, poly4.SignificantCoeffCount());
            Assert.AreEqual("FFFF", poly4.ToString());
            Assert.AreEqual(-1L, encoder.DecodeInt64(poly4));

            var poly5 = encoder.Encode(-2L);
            Assert.AreEqual(2, poly5.SignificantCoeffCount());
            Assert.AreEqual("FFFFx^1 + 1", poly5.ToString());
            Assert.AreEqual(-2L, encoder.DecodeInt64(poly5));

            var poly6 = encoder.Encode(-3L);
            Assert.AreEqual(2, poly6.SignificantCoeffCount());
            Assert.AreEqual("FFFFx^1", poly6.ToString());
            Assert.AreEqual(-3L, encoder.DecodeInt64(poly6));

            var poly7 = encoder.Encode(-0x2671L);
            Assert.AreEqual(9, poly7.SignificantCoeffCount());
            for (int i = 0; i < 9; ++i)
            {
                Assert.AreEqual(0xFFFFUL, poly7[i]);
            }
            Assert.AreEqual(-0x2671L, encoder.DecodeInt64(poly7));

            var poly8 = encoder.Encode(-4374L);
            Assert.AreEqual(9, poly8.SignificantCoeffCount());
            Assert.AreEqual(0xFFFFUL, poly8[8]);
            Assert.AreEqual(1UL, poly8[7]);
            for (int i = 0; i < 7; ++i)
            {
                Assert.IsTrue(poly8[i] == 0);
            }
            Assert.AreEqual(-4374L, encoder.DecodeInt64(poly8));

            var poly9 = encoder.Encode(-0xD4EBL);
            Assert.AreEqual(11, poly9.SignificantCoeffCount());
            for (int i = 0; i < 11; ++i)
            {
                if (i % 3 == 1)
                {
                    Assert.AreEqual(0xFFFFUL, poly9[i]);
                }
                else if (i % 3 == 0)
                {
                    Assert.IsTrue(poly9[i] == 0);
                }
                else
                {
                    Assert.AreEqual(1UL, poly9[i]);
                }
            }
            Assert.AreEqual(-0xD4EBL, encoder.DecodeInt64(poly9));

            var poly10 = encoder.Encode(-30724L);
            Assert.AreEqual(11, poly10.SignificantCoeffCount());
            Assert.AreEqual(0xFFFFUL, poly10[10]);
            Assert.AreEqual(1UL, poly10[9]);
            Assert.AreEqual(1UL, poly10[8]);
            Assert.AreEqual(1UL, poly10[7]);
            Assert.IsTrue(poly10[6] == 0);
            Assert.IsTrue(poly10[5] == 0);
            Assert.AreEqual(0xFFFFUL, poly10[4]);
            Assert.AreEqual(0xFFFFUL, poly10[3]);
            Assert.IsTrue(poly10[2] == 0);
            Assert.AreEqual(1UL, poly10[1]);
            Assert.AreEqual(0xFFFFUL, poly10[0]);
            Assert.AreEqual(-30724L, encoder.DecodeInt64(poly10));

            var encoder2 = new BalancedEncoder(modulus, 13, MemoryPoolHandle.New());
            var poly11 = encoder2.Encode(-126375543984L);
            Assert.AreEqual(11, poly11.SignificantCoeffCount());
            Assert.AreEqual(0xFFFFUL, poly11[10]);
            Assert.AreEqual(1UL, poly11[9]);
            Assert.AreEqual(1UL, poly11[8]);
            Assert.AreEqual(1UL, poly11[7]);
            Assert.IsTrue(poly11[6] == 0);
            Assert.IsTrue(poly11[5] == 0);
            Assert.AreEqual(0xFFFFUL, poly11[4]);
            Assert.AreEqual(0xFFFFUL, poly11[3]);
            Assert.IsTrue(poly11[2] == 0);
            Assert.AreEqual(1UL, poly11[1]);
            Assert.AreEqual(0xFFFFUL, poly11[0]);
            Assert.AreEqual(-126375543984L, encoder2.DecodeInt64(poly11));

            modulus.Set(0xFFFF);
            var encoder3 = new BalancedEncoder(modulus, 7, MemoryPoolHandle.New());
            var poly12 = new Plaintext(6);
            poly12[0] = 1;
            poly12[1] = 0xFFFE; // -1
            poly12[2] = 0xFFFD; // -2
            poly12[3] = 0x8000; // -32767
            poly12[4] = 0x7FFF; // 32767
            poly12[5] = 0x7FFE; // 32766
            Assert.AreEqual(1L + -1 * 7 + -2 * 49 + -32767 * 343 + 32767 * 2401 + 32766 * 16807, encoder3.DecodeInt64(poly12));

            var encoder4 = new BalancedEncoder(modulus, 6, MemoryPoolHandle.New());
            poly8 = new Plaintext(4);
            poly8[0] = 5;
            poly8[1] = 4;
            poly8[2] = 3;
            poly8[3] = (modulus.Value - 2);
            Int64 value = 5 + 4 * 6 + 3 * 36 - 2 * 216;
            Assert.AreEqual(value, encoder4.DecodeInt64(poly8));

            var encoder5 = new BalancedEncoder(modulus, 10, MemoryPoolHandle.New());
            poly9 = new Plaintext(4);
            poly9[0] = 1;
            poly9[1] = 2;
            poly9[2] = 3;
            poly9[3] = 4;
            value = 4321;
            Assert.AreEqual(value, encoder5.DecodeInt64(poly9));

            value = -1234;
            poly10.Set(encoder3.Encode(value));
            Assert.AreEqual(5, poly10.SignificantCoeffCount());
            Assert.IsTrue(value.Equals(encoder3.DecodeInt64(poly10)));

            value = -1234;
            poly11.Set(encoder4.Encode(value));
            Assert.AreEqual(5, poly11.SignificantCoeffCount());
            Assert.IsTrue(value.Equals(encoder4.DecodeInt64(poly11)));

            value = -1234;
            poly12.Set(encoder5.Encode(value));
            Assert.AreEqual(4, poly12.SignificantCoeffCount());
            Assert.IsTrue(value.Equals(encoder5.DecodeInt64(poly12)));
        }

        [TestMethod]
        public void BalancedEncodeDecodeInt32NET()
        {
            var modulus = new SmallModulus(0x10000);
            var encoder = new BalancedEncoder(modulus, 3, MemoryPoolHandle.New());

            var poly = encoder.Encode(0);
            Assert.AreEqual(0, poly.SignificantCoeffCount());
            Assert.IsTrue(poly.IsZero);
            Assert.AreEqual(0, encoder.DecodeInt32(poly));

            var poly1 = encoder.Encode(1);
            Assert.AreEqual(1, poly1.SignificantCoeffCount());
            Assert.AreEqual("1", poly1.ToString());
            Assert.AreEqual(1, encoder.DecodeInt32(poly1));

            var poly2 = encoder.Encode(2);
            Assert.AreEqual(2, poly2.SignificantCoeffCount());
            Assert.AreEqual("1x^1 + FFFF", poly2.ToString());
            Assert.AreEqual(2, encoder.DecodeInt32(poly2));

            var poly3 = encoder.Encode(3);
            Assert.AreEqual(2, poly3.SignificantCoeffCount());
            Assert.AreEqual("1x^1", poly3.ToString());
            Assert.AreEqual(3, encoder.DecodeInt32(poly3));

            var poly4 = encoder.Encode(-1);
            Assert.AreEqual(1, poly4.SignificantCoeffCount());
            Assert.AreEqual("FFFF", poly4.ToString());
            Assert.AreEqual(-1, encoder.DecodeInt32(poly4));

            var poly5 = encoder.Encode(-2);
            Assert.AreEqual(2, poly5.SignificantCoeffCount());
            Assert.AreEqual("FFFFx^1 + 1", poly5.ToString());
            Assert.AreEqual(-2, encoder.DecodeInt32(poly5));

            var poly6 = encoder.Encode(-3);
            Assert.AreEqual(2, poly6.SignificantCoeffCount());
            Assert.AreEqual("FFFFx^1", poly6.ToString());
            Assert.AreEqual(-3, encoder.DecodeInt32(poly6));

            var poly7 = encoder.Encode(-0x2671);
            Assert.AreEqual(9, poly7.SignificantCoeffCount());
            for (int i = 0; i < 9; ++i)
            {
                Assert.AreEqual(0xFFFFUL, poly7[i]);
            }
            Assert.AreEqual(-0x2671, encoder.DecodeInt32(poly7));

            var poly8 = encoder.Encode(-4374);
            Assert.AreEqual(9, poly8.SignificantCoeffCount());
            Assert.AreEqual(0xFFFFUL, poly8[8]);
            Assert.AreEqual(1UL, poly8[7]);
            for (int i = 0; i < 7; ++i)
            {
                Assert.IsTrue(poly8[i] == 0);
            }
            Assert.AreEqual(-4374, encoder.DecodeInt32(poly8));

            var poly9 = encoder.Encode(-0xD4EB);
            Assert.AreEqual(11, poly9.SignificantCoeffCount());
            for (int i = 0; i < 11; ++i)
            {
                if (i % 3 == 1)
                {
                    Assert.AreEqual(0xFFFFUL, poly9[i]);
                }
                else if (i % 3 == 0)
                {
                    Assert.IsTrue(poly9[i] == 0);
                }
                else
                {
                    Assert.AreEqual(1UL, poly9[i]);
                }
            }
            Assert.AreEqual(-0xD4EB, encoder.DecodeInt32(poly9));

            var poly10 = encoder.Encode(-30724);
            Assert.AreEqual(11, poly10.SignificantCoeffCount());
            Assert.AreEqual(0xFFFFUL, poly10[10]);
            Assert.AreEqual(1UL, poly10[9]);
            Assert.AreEqual(1UL, poly10[8]);
            Assert.AreEqual(1UL, poly10[7]);
            Assert.IsTrue(poly10[6] == 0);
            Assert.IsTrue(poly10[5] == 0);
            Assert.AreEqual(0xFFFFUL, poly10[4]);
            Assert.AreEqual(0xFFFFUL, poly10[3]);
            Assert.IsTrue(poly10[2] == 0);
            Assert.AreEqual(1UL, poly10[1]);
            Assert.AreEqual(0xFFFFUL, poly10[0]);
            Assert.AreEqual(-30724, encoder.DecodeInt32(poly10));

            modulus.Set(0xFFFF);
            var encoder2 = new BalancedEncoder(modulus, 7, MemoryPoolHandle.New());
            var poly12 = new Plaintext(6);
            poly12[0] = 1;
            poly12[1] = 0xFFFE; // -1
            poly12[2] = 0xFFFD; // -2
            poly12[3] = 0x8000; // -32767
            poly12[4] = 0x7FFF; // 32767
            poly12[5] = 0x7FFE; // 32766
            Assert.AreEqual(1 + -1 * 7 + -2 * 49 + -32767 * 343 + 32767 * 2401 + 32766 * 16807, encoder2.DecodeInt32(poly12));

            var encoder4 = new BalancedEncoder(modulus, 6, MemoryPoolHandle.New());
            poly8 = new Plaintext(4);
            poly8[0] = 5;
            poly8[1] = 4;
            poly8[2] = 3;
            poly8[3] = (modulus.Value - 2);
            Int32 value = 5 + 4 * 6 + 3 * 36 - 2 * 216;
            Assert.AreEqual(value, encoder4.DecodeInt32(poly8));

            var encoder5 = new BalancedEncoder(modulus, 10, MemoryPoolHandle.New());
            poly9 = new Plaintext(4);
            poly9[0] = 1;
            poly9[1] = 2;
            poly9[2] = 3;
            poly9[3] = 4;
            value = 4321;
            Assert.AreEqual(value, encoder5.DecodeInt32(poly9));

            value = -1234;
            poly10.Set(encoder2.Encode(value));
            Assert.AreEqual(5, poly10.SignificantCoeffCount());
            Assert.IsTrue(value.Equals(encoder2.DecodeInt32(poly10)));

            value = -1234;
            var poly11 = encoder4.Encode(value);
            Assert.AreEqual(5, poly11.SignificantCoeffCount());
            Assert.AreEqual(value, encoder4.DecodeInt32(poly11));

            value = -1234;
            poly12.Set(encoder5.Encode(value));
            Assert.AreEqual(4, poly12.SignificantCoeffCount());
            Assert.IsTrue(value.Equals(encoder5.DecodeInt32(poly12)));
        }

        [TestMethod]
        public void BinaryFractionalEncodeDecodeNET()
        {
            var polyModulus = new BigPoly("1x^1024 + 1");
            var modulus = new SmallModulus(0x10000);
            var encoder = new BinaryFractionalEncoder(modulus, polyModulus, 500, 50, MemoryPoolHandle.New());

            var poly = encoder.Encode(0.0);
            Assert.IsTrue(poly.IsZero);
            Assert.AreEqual(0.0, encoder.Decode(poly));
            
            var poly1 = encoder.Encode(-1.0);
            Assert.AreEqual(-1.0, encoder.Decode(poly1));

            var poly2 = encoder.Encode(0.1);
            Assert.IsTrue(Math.Abs(encoder.Decode(poly2)-0.1)/0.1 < 0.000001);
            
            var poly3 = encoder.Encode(3.123);
            Assert.IsTrue(Math.Abs(encoder.Decode(poly3) - 3.123) / 3.123 < 0.000001);

            var poly4 = encoder.Encode(-123.456);
            Assert.IsTrue(Math.Abs(encoder.Decode(poly4) + 123.456) / (-123.456) < 0.000001);

            var poly5 = encoder.Encode(12345.98765);
            Assert.IsTrue(Math.Abs(encoder.Decode(poly5) - 12345.98765) / 12345.98765 < 0.000001);
        }

        [TestMethod]
        public void BalancedFractionalEncodeDecodeNET()
        {
            var polyModulus = new BigPoly("1x^1024 + 1");
            var modulus = new SmallModulus(0x10000);

            var poly = new Plaintext();
            var poly1 = new Plaintext();
            var poly2 = new Plaintext();
            var poly3 = new Plaintext();
            var poly4 = new Plaintext();
            var poly5 = new Plaintext();

            for(ulong b=3; b<20; b+=2)
            {
                var encoder = new BalancedFractionalEncoder(modulus, polyModulus, 500, 50, b, MemoryPoolHandle.New());

                poly = encoder.Encode(0.0);
                Assert.AreEqual(1, poly.CoeffCount);
                Assert.IsTrue(poly.IsZero);
                Assert.AreEqual(0.0, encoder.Decode(poly));

                poly1 = encoder.Encode(-1.0);
                Assert.AreEqual(-1.0, encoder.Decode(poly1));

                poly2 = encoder.Encode(0.1);
                Assert.IsTrue(Math.Abs(encoder.Decode(poly2) - 0.1) / 0.1 < 0.000001);

                poly3 = encoder.Encode(3.123);
                Assert.IsTrue(Math.Abs(encoder.Decode(poly3) - 3.123) / 3.123 < 0.000001);

                poly4 = encoder.Encode(-123.456);
                Assert.IsTrue(Math.Abs(encoder.Decode(poly4) + 123.456) / (-123.456) < 0.000001);

                poly5 = encoder.Encode(12345.98765);
                Assert.IsTrue(Math.Abs(encoder.Decode(poly5) - 12345.98765) / 12345.98765 < 0.000001);
            }
        }
    }
}
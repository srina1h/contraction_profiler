import unittest
from einstein_notation_conversion import construct_einstein_notation

class TestEinsteinNotation(unittest.TestCase):
    def test_einstein_notation_1(self):
        self.assertEqual(construct_einstein_notation(3, 3, ([-1], [-1])), 'abc * dec -> abde')

    def test_einstein_notation_2(self):
        self.assertEqual(construct_einstein_notation(4, 2, ([-1], [0])), 'abcd * de -> abce')

    def test_einstein_notation_3(self):
        self.assertEqual(construct_einstein_notation(4, 4, ([0, -1], [0, -1])), 'abcd * aefd -> bcef')

    def test_einstein_notation_4(self):
        self.assertEqual(construct_einstein_notation(3, 3, ([0, 1], [0, 1])), 'abc * abd -> cd')

    def test_einstein_notation_5(self):
        self.assertEqual(construct_einstein_notation(3, 3, ([0], [-1])), 'abc * dea -> bcde')
    

if __name__ == '__main__':
    unittest.main()
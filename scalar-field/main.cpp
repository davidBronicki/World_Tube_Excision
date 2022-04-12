#include <iostream>
#include <vector>
#include <math.h>

using std::cout;
using std::endl;

template <typename T>
class Chebyshev
{
	std::vector<std::vector<T>> coefs;

	// std::vector<std::vector<T>> zeros;
	// std::vector<std::vector<T>> extrema;

	//extrema of order+1th polynomial
	std::vector<T> collocationPoints;
	public:
	Chebyshev(int order):
		coefs(order + 1),
		zeros(order + 1),
		extrema(order + 1),
		collocationPoints(order + 1)
	{
		coefs[0].push_back(1);
		if (order == 0)
		{
			return;
		}
		coefs[1].push_back(1);
		if (order == 1)
		{
			return;
		}
		for (int n = 2; n <= order; ++n)
		{
			coefs[n].resize((n + 1) / 2);
			if (n % 2 == 0)
			{
				coefs[n][0] = -coefs[n - 2][0];
				for (int i = 1; i < n / 2; ++i)
				{
					coefs[n][i] = 2 * coefs[n - 1][i - 1] - coefs[n - 2][i];
				}
				coefs[n][n / 2] = 2 * coefs[n - 1][n / 2 - 1];
			}
			else
			{
				for (int i = 0; i < n / 2; ++i)
				{
					coefs[n][i] = 2 * coefs[n - 1][i] - coefs[n - 2][i];
				}
				coefs[n][n / 2] = 2 * coefs[n - 1][n / 2];
			}
		}

		// for (int n = 0; n <= order; ++n)
		// {
		// 	zeros[n].resize(n);
		// 	extrema[n].resize(n + 1)
		// 	for (int i )
		// }
		for (n = order; n >= 0; --n)
		{
			collocationPoints[n] = cos(M_PI * (n + 0.5) / (order + 1));
		}
	}

	T evaluate(int n, T x)
	{
		T monomialValue = (n % 2 == 0)?
			1:
			x;
		T xSquared = x * x;
		T output = monomialValue * coefs[n][0];
		for (int i = 1; i < (n + 2) / 2; ++i)
		{
			monomialValue *= xSquared;
			output += monomialValue * coefs[n][i];
		}
		return output;
	}
};

int main()
{
	Chebyshev<double> test(5);
}

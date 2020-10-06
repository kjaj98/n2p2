// n2p2 - A neural network potential package
// Copyright (C) 2018 Andreas Singraber (University of Vienna)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#ifndef COREFUNCTION_H
#define COREFUNCTION_H

namespace nnp
{

class CoreFunction
{
public:
    /// List of available function types.
    enum class Type
    {
        /** @f$f(x) = (2x - 3)x^2 + 1@f$
         */
        POLY1,
        /** @f$f(x) = ((15 - 6x)x - 10) x^3 + 1@f$
         */
        POLY2,
        /** @f$f(x) = (x(x(20x - 70) + 84) - 35)x^4 + 1@f$
         */
        POLY3,
        /** @f$f(x) = (x(x((315 - 70x)x - 540) + 420) - 126)x^5 + 1@f$
         */
        POLY4
    };

    /** Constructor, initializes to #Type::POLY2.
     */
    CoreFunction();
    /** Set function type.
     *
     * @param[in] type Type of core function used.
     */
    void   setType(Type const type);
    /** Getter for #type.
     *
     * @return Type used.
     */
    Type   getType() const;
    /** Calculate function value @f$f(x)@f$.
     *
     * @param[in] x Function argument.
     * @return Function value.
     */
    double f(double x) const;
    /** Calculate derivative of function at argument @f$\frac{df(x)}{dx}@f$.
     *
     * @param[in] x Function argument.
     * @return Function derivative value.
     */
    double df(double x) const;
    /** Calculate function and derivative at once.
     *
     * @param[in] x Function argument.
     * @param[out] fx Function value.
     * @param[out] dfx Derivative value.
     */
    void   fdf(double x, double& fx, double& dfx) const;

private:
    /// Core function type.
    Type       type;
    /// Function pointer to f.
    double     (CoreFunction::*fPtr)(double x) const;
    /// Function pointer to df.
    double     (CoreFunction::*dfPtr)(double x) const;
    /// Function pointer to fdf.
    void       (CoreFunction::*fdfPtr)(double  x,
                                       double& fx,
                                       double& dfx) const;

    // Individual functions.
    double   fPOLY1(double x) const;
    double  dfPOLY1(double x) const;
    void   fdfPOLY1(double x, double& fx, double& dfx) const;

    double   fPOLY2(double x) const;
    double  dfPOLY2(double x) const;
    void   fdfPOLY2(double x, double& fx, double& dfx) const;

    double   fPOLY3(double x) const;
    double  dfPOLY3(double x) const;
    void   fdfPOLY3(double x, double& fx, double& dfx) const;

    double   fPOLY4(double x) const;
    double  dfPOLY4(double x) const;
    void   fdfPOLY4(double x, double& fx, double& dfx) const;

    double   fPOLYA(double x) const;
    double  dfPOLYA(double x) const;
    void   fdfPOLYA(double x, double& fx, double& dfx) const;
};

//////////////////////////////////
// Inlined function definitions //
//////////////////////////////////

inline CoreFunction::Type CoreFunction::getType() const
{
    return type;
}

inline double CoreFunction::f(double x) const
{
    return (this->*fPtr)(x);
}

inline double CoreFunction::df(double x) const
{
    return (this->*dfPtr)(x);
}

inline void CoreFunction::fdf(double x, double& fx, double& dfx) const
{
    (this->*fdfPtr)(x, fx, dfx);
    return;
}

inline double CoreFunction::fPOLY1(double x) const
{
    return (2.0 * x - 3.0) * x * x + 1.0;
}

inline double CoreFunction::dfPOLY1(double x) const
{
    return x * (6.0 * x - 6.0);
}

inline void CoreFunction::fdfPOLY1(double x, double& fx, double& dfx) const
{
    fx = (2.0 * x - 3.0) * x * x + 1.0;
    dfx = x * (6.0 * x - 6.0);
    return;
}

inline double CoreFunction::fPOLY2(double x) const
{
    return ((15.0 - 6.0 * x) * x - 10.0) * x * x * x + 1.0;
}

inline double CoreFunction::dfPOLY2(double x) const
{
    return x * x * ((60.0 - 30.0 * x) * x - 30.0);
}

inline void CoreFunction::fdfPOLY2(double x, double& fx, double& dfx) const
{
    double const x2 = x * x;
    fx = ((15.0 - 6.0 * x) * x - 10.0) * x * x2 + 1.0;
    dfx = x2 * ((60.0 - 30.0 * x) * x - 30.0);
    return;
}

inline double CoreFunction::fPOLY3(double x) const
{
    double const x2 = x * x;
    return (x * (x * (20.0 * x - 70.0) + 84.0) - 35.0) * x2 * x2 + 1.0;
}

inline double CoreFunction::dfPOLY3(double x) const
{
    return x * x * x * (x * (x * (140.0 * x - 420.0) + 420.0) - 140.0);
}

inline void CoreFunction::fdfPOLY3(double x, double& fx, double& dfx) const
{
    double const x2 = x * x;
    fx = (x * (x * (20.0 * x - 70.0) + 84.0) - 35.0) * x2 * x2 + 1.0;
    dfx = x2 * x * (x * (x * (140.0 * x - 420.0) + 420.0) - 140.0);
    return;
}

inline double CoreFunction::fPOLY4(double x) const
{
    double const x2 = x * x;
    return (x * (x * ((315.0 - 70.0 * x) * x - 540.0) + 420.0) - 126.0) *
           x2 * x2 * x + 1.0;
}

inline double CoreFunction::dfPOLY4(double x) const
{
    double const x2 = x * x;
    return x2 * x2 *
           (x * (x * ((2520.0 - 630.0 * x) * x - 3780.0) + 2520.0) - 630.0);
}

inline void CoreFunction::fdfPOLY4(double x, double& fx, double& dfx) const
{
    double x4 = x * x;
    x4 *= x4;
    fx = (x * (x * ((315.0 - 70.0 * x) * x - 540.0) + 420.0) - 126.0) *
         x * x4 + 1.0;
    dfx = x4 *
          (x * (x * ((2520.0 - 630.0 * x) * x - 3780.0) + 2520.0) - 630.0);
    return;
}

//inline double CoreFunction::fPOLYA(double x) const
//{
//    double const p = 2.0 * x - x * x;
//    return ((15.0 - 6.0 * p) * p - 10.0) * p * p * p + 1.0;
//}
//
//inline double CoreFunction::dfPOLYA(double x) const
//{
//    double const p  = 2.0 * x - x * x;
//    double const dp =     2.0 - 2.0 * x;
//    double const dx = p * p * ((60.0 - 30.0 * p) * p - 30.0);
//    return dx * dp;
//}
//
//inline void CoreFunction::fdfPOLYA(double x, double& fx, double& dfx) const
//{
//    double const p  = 2.0 * x - x * x;
//    double const p2 = p * p;
//    double const dp = 2.0 - 2.0 * x;
//    double const dx = p2 * ((60.0 - 30.0 * p) * p - 30.0);
//    fx = ((15.0 - 6.0 * p) * p - 10.0) * p * p2 + 1.0;
//    dfx = dx * dp;
//    return;
//}

}

#endif

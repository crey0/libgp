// libgp - Gaussian process library for Machine Learning
#include "cov_exp_ard.h"
#include <cmath>

double mmod(double a, int n)
{
  return a - floor(a/n) * n;
}

namespace libgp
{
  
  bool CovExpArd::init(int n)
  {
    input_dim = n;
    param_dim = n+1;
    ell.resize(input_dim);
    loghyper.resize(param_dim);
    return true;
  }
  
  double CovExpArd::get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2)
  {  
    double z = distance(x1, x2).cwiseQuotient(ell).norm();
    return sf2*exp(-z);
  }
  
  void CovExpArd::grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad)
  {
    auto d = distance(x1, x2).array().square();
    double z = sqrt(d.sum());
    double k = sf2*exp(-z);
    grad.head(input_dim) = ell.array().pow(-3).cwiseProduct(d) / z * k;
    grad(input_dim) = 2.0 * k;
  }
  
  void CovExpArd::set_loghyper(const Eigen::VectorXd &p)
  {
    CovarianceFunction::set_loghyper(p);
    for(size_t i = 0; i < input_dim; ++i) ell(i) = exp(loghyper(i));
    sf2 = exp(2*loghyper(input_dim));
  }

  Eigen::VectorXd CovExpArd::distance(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2)
  {
    return (x1-x2);
  }
  
  std::string CovExpArd::to_string()
  {
    return "CovExpArd";
  }


  void CovExpArdPhi::setPhi(int phi)
  {
    this->phi = phi;
  }

  int CovExpArdPhi::getPhi()
  {
    return phi;
  }

  Eigen::VectorXd CovExpArdPhi::distance(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2)
  {
    Eigen::VectorXd d = (x1-x2);
    d(phi) = mmod(d(phi) + 180, 360) - 180;
    return d;
  }
  
  std::string CovExpArdPhi::to_string()
  {
    return "CovExpArdPhi";
  }
}


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
    Eigen::VectorXd d(input_dim);
    componentwise_distance(x1,x2,d);
    double z = d.cwiseQuotient(ell).norm();
    return sf2*exp(-z);
  }
  
  void CovExpArd::grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad)
  {
    Eigen::VectorXd dist(input_dim);
    componentwise_distance(x1, x2, dist);
    auto d = dist.array().square();
    double z = sqrt(d.sum());
    double k = sf2*exp(-z);
    if (z==0) // avoid segfault when z==0, grad should be 0
    {
      z=1;
    }
    grad.head(input_dim) = ell.array().pow(-3).cwiseProduct(d) / z * k;
    grad(input_dim) = 2.0 * k;
  }
  
  void CovExpArd::set_loghyper(const Eigen::VectorXd &p)
  {
    CovarianceFunction::set_loghyper(p);
    for(size_t i = 0; i < input_dim; ++i) ell(i) = exp(loghyper(i));
    sf2 = exp(2*loghyper(input_dim));
  }
  
  std::string CovExpArd::to_string()
  {
    return "CovExpArd";
  }

  /**
     CovExpArdphi
   **/
  
  void CovExpArdPhi::setPhi(int phi)
  {
    this->phi = phi;
  }

  int CovExpArdPhi::getPhi()
  {
    return phi;
  }

  void CovExpArdPhi::componentwise_distance(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &d)
  {
    d = (x1-x2);
    d(phi) = mmod(d(phi) + 180, 360) - 180;
  }
  
  std::string CovExpArdPhi::to_string()
  {
    return "CovExpArdPhi";
  }
}


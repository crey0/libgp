#ifndef __COV_EXP_ARD_H__
#define __COV_EXP_ARD_H__

#include "cov.h"

namespace libgp
{
  
  /** Exponential covariance function with automatic relevance detection.
   *  Computes the exponential covariance
   *  \f$k_{EXP}(x, y) := \alpha^2 \exp(-(x-y)^T\Lambda^{-1}(x-y))\f$,
   *  with \f$\Lambda = diag(l_1^2, \dots, l_n^2)\f$ being the characteristic
   *  length scales and \f$\alpha\f$ describing the variability of the latent
   *  function. The parameters \f$l_1^2, \dots, l_n^2, \alpha\f$ are expected
   *  in this order in the parameter array.
   *  @ingroup cov_group
   *  @author 
   */
  class CovExpArd : public CovarianceFunction
  {
  public:
    CovExpArd () {}
    virtual ~CovExpArd (){}
    bool init(int n);
    double get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2);
    void grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad);
    void set_loghyper(const Eigen::VectorXd &p);
    virtual Eigen::VectorXd distance(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2){return x1 - x2;}
    virtual std::string to_string();
  private:
    Eigen::VectorXd ell;
    double sf2;
  };

  class CovExpArdPhi : public CovExpArd
  {
  public:
    CovExpArdPhi () : phi(2) {}
    virtual ~CovExpArdPhi () {}
    virtual Eigen::VectorXd distance(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2);
    virtual std::string to_string();
    int getPhi();
    void setPhi(int phi);
  private:
    int phi;
  };
  
}

#endif /* __COV_EXP_ARD_H__ */

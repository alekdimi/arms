import scipy.stats
import numpy as np
import matplotlib.pyplot as plt


class DirichletCopula:
    def __init__(self, dim):
        self.dim = dim
        self.alpha = np.ones(dim)

    def sample(self, size=None, both=True):
        unif = np.random.uniform(size=tuple(size) + (self.dim, ))
        exp = -np.log(unif)
        dirichlet = exp / exp.sum(axis=-1, keepdims=True) 
        u = np.power(1 - dirichlet, self.dim - 1)
        return (u, 1 - u) if both else u

    def bivariate_cdf(self, p, q):
        term = np.power(p, 1 / (self.dim - 1)) + np.power(q, 1 / (self.dim - 1)) - 1
        return np.power(np.maximum(term, 0), self.dim - 1)

    def debias(self, p, inverted=False):
        prob = (1 - inverted) * p + inverted * (1 - p)
        return p * (1 - p) / (prob - self.bivariate_cdf(prob, prob))


class GaussianCopula:
    def __init__(self, dim):
        self.dim = dim
        self.cov = np.ones([dim, dim]) * 1 / (1 - dim) + np.eye(dim) * dim / (dim - 1)
        self.mvn = scipy.stats.multivariate_normal(
            [0] * dim, self.cov, allow_singular=True)
        self.bvn = scipy.stats.multivariate_normal(mean=[0, 0], allow_singular=True,
                                                   cov=np.array([[1.0, 1 / (1 - dim)], [1 / (1 - dim), 1.0]]))
        self.cdf = scipy.stats.norm.cdf
        self.inverse_cdf = scipy.stats.norm.ppf

    def sample(self, size, uniform=True):
        normals = self.mvn.rvs(size)
        return self.cdf(normals) if uniform else normals

    def bivariate_cdf(self, p, q):
        return self.bvn.cdf(np.stack([scipy.stats.norm.ppf(p), scipy.stats.norm.ppf(q)]).T)

    def debias(self, p):
        return p * (1 - p) / (p - self.bivariate_cdf(p, p))

sigmoid = lambda x: 1 / (1 + np.exp(-x))

def train(method, num_samples=6, steps=None, p0=0.49, num_mc=500, p_init=0.05):
    f = lambda z: (z - p0) ** 2
    N = num_samples // 2 if method[-3:] == 'ARM' or 'DisARM' in method else num_samples
    if N > 1:
      gaussian_copula = GaussianCopula(N) 
      dirichlet_copula = DirichletCopula(N)  
    
    if steps is not None:
        grads, varss, logits = np.zeros(steps), np.zeros(steps), np.zeros(steps + 1) + np.log(p_init)
    else:
        grads, varss, logits = np.array([]), np.array([]), np.array([np.log(p_init)])
    i = 0
    while True:
        p = sigmoid(logits[i])
        if steps is not None and i >= steps:
            break
        if steps is None and p >= (1 - p_init):
            break
        
        u = np.random.uniform(size=[num_mc, N])
        z, zp = u < p, (1 - u) < p

        if method == 'True':
            grad = ((1 - 2 * p0) * p * (1 - p)).reshape([1, 1])
            
        elif method == 'ARMS-N':
            gg = gaussian_copula.sample(num_mc, uniform=False)
            z = 1 * (gg < gaussian_copula.inverse_cdf(p))
            debias = gaussian_copula.debias(p)
            grad = (f(z) - f(z).mean(axis=1, keepdims=True)) * (z - p) * N / (N - 1) * debias
            
        elif method == 'ARM':
            grad = (f(zp) - f(z)) * (u - 0.5)
            
        elif method == 'DisARM':
            grad = 0.5 * (f(zp) - f(z)) * np.maximum(p, 1 - p) * ((1 - u < p) * 1.0 - (u < p))      
            
        elif method == 'LOORF':
            grad = (f(z) - f(z).mean(axis=1, keepdims=True)) * (z - p) * N / (N - 1)
            
        elif method == 'ARMS-D':
            uc = dirichlet_copula.sample(size=[num_mc], both=False)
            zc, zcp = uc < p, (1 - uc) < p
            debias, debias_inv = dirichlet_copula.debias(p), dirichlet_copula.debias(p, inverted=True)
            if p < 0.5:
                grad = (f(zc) - f(zc).mean(axis=1, keepdims=True)) * (zc - p) * N / (N - 1) * debias
            else:
                grad = (f(zcp) - f(zcp).mean(axis=1, keepdims=True)) * (zcp - p) * N / (N - 1) * debias_inv
        else:
            raise('Unknown method.')

        grad_est = grad.mean(axis=1)  
        update = grad_est.mean()
        if steps is not None:
            grads[i] = grad_est.mean()
            varss[i] = grad_est.var()
            logits[i + 1] = logits[i] + update
        else:
            grads  = np.append(grads, grad_est.mean())
            varss  = np.append(varss, grad_est.var())
            logits = np.append(logits, logits[i] + update)
            
        i += 1
    return grads, varss, logits


if __name__ == '__main__':
  p0 = 0.49
  Ns = [2, 4, 6, 8, 10]
  p_init = 0.1

  plt.style.use('seaborn-whitegrid')
  cmap = plt.get_cmap('Dark2')
  methods = ['True', 'ARMS-D', 'ARMS-N', 'ARM', 'DisARM', 'LOORF']
  zorders = {'ARMS-D': 10, 'ARMS-N': 8, 'LOORF': 6, 'DisARM': 4, 'ARM': 2}
  method_colors = {'ARMS-D': cmap(0), 'ARMS-N': cmap(4), 'LOORF': cmap(2), 'DisARM': cmap(1), 'ARM': cmap(3)}
                 
  grads, varss, logits = {}, {}, {}
  for N in Ns:
      for method in methods:
          g, v, l = train(method, num_samples=N, p0=p0, p_init=p_init, num_mc=1000)
          grads[(N, method)], varss[(N, method)], logits[(N, method)] = g, v, l
          
  fontsize = 12
  _, ax = plt.subplots(1, len(Ns), figsize=(0.8 * 3.4 * len(Ns), 0.7 * 2))
  for i in range(len(Ns)):
      N = Ns[i]
      for method in methods[1:]:
          probs = sigmoid(logits[(N, method)])[1:]
          ax[i].plot(probs, varss[(N, method)], label=method, alpha=0.8, 
                    lw=0.5, color=method_colors[method], zorder=zorders[method])
      ax[i].set_xlabel('$\sigma(\phi)$')
      ax[i].set_title(str(N) + ' samples')
      ax[i].ticklabel_format(style='sci', scilimits=(-2, 2))
      for spine in ['left', 'right', 'top', 'bottom']:
          ax[i].spines[spine].set_visible(False)    
  ax[0].set_ylabel('Gradient variance', fontsize=fontsize + 1)

  ax[-1].legend(fontsize=fontsize + 2)
  leg = ax[-1].legend()
  for line in leg.get_lines():
      line.set_linewidth(3.0) 
  bb = leg.get_bbox_to_anchor().inverse_transformed(ax[-1].transAxes)
  bb.x0 += 1.0
  bb.x1 += 0.0
  leg.set_bbox_to_anchor(bb, transform = ax[-1].transAxes)
  plt.show()
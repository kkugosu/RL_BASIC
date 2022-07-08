in sac, we add entropy term to reward so that policy follows boltzmann distribution.
by adding entropy term, algorithm become more robust to distributional shift problem, but not perfectly prevent.

this algorithm guarantee improvement of Q function but improvement of Q function is not always improve overall performance.
to improve overall performance, we restrict the amount of update. such as kld which is used in trpo.

below is formula of dV, dQ, d$\pi$

f means reparameterization trick

$$\nabla_\phi J_\pi (\phi) = \nabla_\phi log \pi_\phi (a_t | s_t) + (\nabla_{at}log \pi_\phi (a_t | s_t) - 
\nabla_{at} Q(s_t,a_t))\nabla_\phi f_\phi (\epsilon_t ; s_t)$$

$$\nabla_\theta J_Q (\theta) = \nabla_\theta Q_\theta (a_t,s_t)(Q_\theta(s_t,a_t) - r(s_t,a_t) - \gamma V_\psi (s_{t+1}))$$

$$\nabla_\psi J_V (\psi) = \nabla_\psi V_\psi (s_t) (V_\psi(s_t) - Q_\theta(s_t,a_t) + log\pi_\psi(a_t | s_t))$$

below is proximal procedure of algorithm

$$ 1.\ take \ action \ a \sim \pi_\theta (a|s), store (s, a, s', r)\ in \ memory $$

$$ 2.\ sample \ a \ batch \ (s_i, a_i, r_i, s_i')\ from \ memory$$

$$ 3.\psi \leftarrow \psi - \lambda_V \nabla_\psi J_V(\psi)$$

$$ 4.\theta \leftarrow \theta - \lambda_Q \nabla_\theta J_Q(\theta)$$

$$ 5.\phi \leftarrow \phi - \lambda_\pi \nabla_\phi J_\pi(\phi)$$

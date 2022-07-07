$$\nabla_\phi J_\pi (\phi) = \nabla_\phi log \pi_\phi (a_t | s_t) + (\nabla_{at}log \pi_\phi (a_t | s_t) - 
\nabla_{at} Q(s_t,a_t))\nabla_\phi f_\phi (\epsilon_t ; s_t)$$

$$\nabla_\theta J_Q (\theta) = \nabla_\theta Q_\theta (a_t,s_t)(Q_\theta(s_t,a_t) - r(s_t,a_t) - \gamma V_\psi (s_{t+1}))$$

$$\nabla_\psi J_V (\psi) = \nabla_\psi V_\psi (s_t) (V_\psi(s_t) - Q_\theta(s_t,a_t) + log\pi_\psi(a_t | s_t))$$

$$\psi \leftarrow \psi - \lambda_V \nabla_\psi J_V(\psi)$$

$$\theta \leftarrow \theta - \lambda_Q \nabla_\theta J_Q(\theta)$$

$$\phi \leftarrow \phi - \lambda_\pi \nabla_\phi J_\pi(\phi)$$

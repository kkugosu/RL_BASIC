## PPO

we don't have to use Q function approximization. because V function and policy function share parameter $\theta$ and we update these functions simultaneously, we always refill buffer with new policy.

$$ L^{CLIP} (\theta) = \hat{E_t} (min (r_t (\theta) \hat{A_t} , clip (r_t (\theta), 1 - \epsilon, 1 + \epsilon)\hat{A_t})$$

$$ L_t^{CLIP + VF + S} (\theta) = \hat{E_t} (L_t^{CLIP} (\theta) - c_1L_t^{VF}(\theta) + c_2S [\pi_\theta] (s_t)) $$

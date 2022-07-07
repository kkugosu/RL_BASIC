## AC

AC is one of policy gradient which use Q function(network) as baseline
we can't use value function as baseline because trajectary in memory used old policy and value function is affected by policy while estimation.

AC use three network. first network is for policy update. 

<img width="647" alt="스크린샷 2022-05-23 오후 8 14 51" src="https://user-images.githubusercontent.com/24292848/169807378-9496b69c-bbec-4a45-ad6f-31b0215797ab.png">

second and third network below is for q update. 

<img width="603" alt="스크린샷 2022-05-27 오전 11 20 44" src="https://user-images.githubusercontent.com/24292848/170615900-dc75e44b-7657-42ba-9787-a00964f3a369.png">

<img width="630" alt="스크린샷 2022-05-27 오전 11 20 51" src="https://user-images.githubusercontent.com/24292848/170615930-ead14e32-7f9e-44bd-88d9-1ed8e20706e0.png">

$$ 1.\ take \ action \ a \sim \pi_\theta (a|s), store (s, a, s', r)\ in \ memory $$

$$ 2.\ sample \ a \ batch \ (s_i, a_i, r_i, s_i')\ from \ memory, \ load \ \phi' \ from \ \phi$$

$$ 3.\ \phi \leftarrow \phi - \alpha \Sigma_i {dQ \over d\phi} (s_i , a_i) (Q_\phi (s_i , a_i) - (r(s_i , a_i) + \gamma Q_{\phi'}(s_i',a_i'))) $$

$$ "you\ can\ pick\ a_i'\ from\ current\ policy\ "$$

$$ 4.\ \nabla_\theta J(\theta) \approx {1 \over N} \Sigma_i \nabla_\theta log \pi_\theta (a_i | s_i) Q_\phi (s_i , a_i) $$

$$ "you\ can\ pick\ a_i\ from\ current\ policy\ "$$

$$ 5.\ \theta \leftarrow \theta + \alpha \nabla_\theta J (\theta) $$

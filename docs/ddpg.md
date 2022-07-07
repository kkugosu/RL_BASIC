
## DDPG

previously i sampled action from continuous space but there are many ways to choose action in continuous space.

cem and cma-es works in low dimensional space

naf assume q value follow gaussian distribution

ddpg directly choose one action which make max q value.

this only different from ac in policy network

<img width="574" alt="스크린샷 2022-05-27 오전 11 41 31" src="https://user-images.githubusercontent.com/24292848/170618175-db7ef51a-6b7e-46c7-ae8e-0fc7e5437d36.png">

$$ 1.\ take \ action \ a \sim \pi_\theta (a|s), store (s, a, s', r)\ in \ memory $$

$$ 2.\ sample \ a \ batch \ (s_i, a_i, r_i, s_i')\ from \ memory, \ load \ \phi' \ from \ \phi$$

$$ 3.\ \phi \leftarrow \phi - \alpha \Sigma_i {dQ \over d\phi} (s_i , a_i) (Q_\phi (s_i , a_i) - (r(s_i , a_i) + \gamma Q_{\phi'}(s_i',a_i'))) $$

$$ "you\ can\ pick\ a_i'\ from\ current\ policy \ \mu' \ "$$

$$ we \ just\ have\ to\ find\ \theta \ which\ maxmize\ Q,\ {dQ_\phi \over d\theta} = {da \over d\theta} {dQ_\phi \over da}$$

$$ 4.\ \theta \leftarrow \theta + \beta \Sigma_j {d\mu \over d\theta} (s_j){dQ_\phi \over da}(s_j, \mu(s_j)) $$

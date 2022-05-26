## PG (with no baseline)

# <img src="https://render.githubusercontent.com/render/math?math=1 \. \ \ sample \ \left\{\tau^ i \right\} from \ \ \pi_\theta \ \left(a_t \ \ | \ \ s_t \right) ">

# <img src="https://render.githubusercontent.com/render/math?math=2 \. \ \ \nabla_\theta J \left( \theta \right) \approx \Sigma_i \ \left( \Sigma_t \nabla_\theta \ log \  \pi_\theta \ \left(a_t^i \ \ | \ \ s_t^i \right) \Sigma_{t'=t}^T r \left(a_{t'}^i \ \ | \ \ s_{t'}^i \right) \right)">

# <img src="https://render.githubusercontent.com/render/math?math=3 \. \ \theta \leftarrow \theta \  \dotplus \alpha \nabla_\theta J \left( \theta \right) ">

<img width="647" alt="스크린샷 2022-05-23 오후 8 14 51" src="https://user-images.githubusercontent.com/24292848/169807378-9496b69c-bbec-4a45-ad6f-31b0215797ab.png">


## DQN (with static network)

$$ 1 \. save\ update \ network\ parameters\ to\ static\ network\ parameters: \phi ' \leftarrow \phi $$
$$ 2 \. collect \ dataset \ ( ( s_i, a_i, s_i ' , r_i ) )  \ using \ some \ policy, \ add \ it\ to\ memory $$
$$ 3 \. \phi \leftarrow \phi - \alpha \Sigma_i {dQ \over d\phi} (s_i , a_i) (Q_\phi (s_i , a_i) - (r(s_i , a_i) + \gamma max_{a'}Q_{\phi'}(s_i',a_i'))) $$


<img width="647" alt="스크린샷 2022-05-23 오후 8 30 01" src="https://user-images.githubusercontent.com/24292848/169809875-e733ff29-249f-43e1-95bc-a9209d0d0ae7.png">

this network represent $ Q_\phi value $

choose a2 and next state become s'. we calculate Q(s'a') from static network

<img width="603" alt="스크린샷 2022-05-23 오후 8 35 29" src="https://user-images.githubusercontent.com/24292848/169810706-73d3b59b-0db0-4176-8243-8b2fb24a0697.png">

this network represent $ Q_\phi' value $

update (nework_to_update) to reduce gap between Q(s,a2)(which is 2), reward(which is 5) + Q(s'a')(which is 5)
this method can't guarantee convergence. because of max operation, this fomula try to converge infinity norm and L2 norm at the same time. this is impossible in general case

## AC

$$ 1.\ take \ action \ a \sim \pi_\theta (a|s), store (s, a, s', r)\ in \ memory $$
$$ 2.\ sample \ a \ batch \ (s_i, a_i, r_i, s_i')\ from \ memory, \ load \ \phi' \ from \ \phi$$
$$ 3.\ \phi \leftarrow \phi - \alpha \Sigma_i {dQ \over d\phi} (s_i , a_i) (Q_\phi (s_i , a_i) - (r(s_i , a_i) + \gamma Q_{\phi'}(s_i',a_i'))) $$
$$ "you\ can\ pick\ a_i'\ from\ memory\ or\ from\ current\ policy\ anything\ is\ okey"$$
$$ 4.\ \nabla_\theta J(\theta) \approx {1 \over N} \Sigma_i \nabla_\theta log \pi_\theta (a_i | s_i) Q_\phi (s_i , a_i) $$
$$ 5.\ \theta \leftarrow \theta + \alpha \nabla_\theta J (\theta) $$

## DDPG

previously i sampled action from continuous space but there are many ways to choose action in continuous space.

cem and cma-es works in low dimensional space

naf assume q value follow gaussian distribution

ddpg directly choose one action which make max q value.

$$ 1.\ take \ action \ a \sim \pi_\theta (a|s), store (s, a, s', r)\ in \ memory $$
$$ 2.\ sample \ a \ batch \ (s_i, a_i, r_i, s_i')\ from \ memory, \ load \ \phi' \ from \ \phi$$
$$ 3.\ \phi \leftarrow \phi - \alpha \Sigma_i {dQ \over d\phi} (s_i , a_i) (Q_\phi (s_i , a_i) - (r(s_i , a_i) + \gamma Q_{\phi'}(s_i',a_i'))) $$
$$ "you\ can\ pick\ a_i'\ from\ memory\ or\ from\ current\ policy \ \mu' \ anything\ is\ okey"$$
$$ 4.\ \theta \leftarrow \theta + \beta \Sigma_j {d\mu \over d\theta} (s_j){dQ_\phi \over da}(s_j, \mu(s_j)) $$
$$ we \ just\ have\ to\ find\ \theta \ which\ maxmize\ Q,\ {dQ_\phi \over d\theta} = {da \over d\theta} {dQ_\phi \over da}$$


## TRPO
## PPO


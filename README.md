## PG (with no baseline)

# <img src="https://render.githubusercontent.com/render/math?math=1 \. \ \ sample \ \left\{\tau^ i \right\} from \ \ \pi_\theta \ \left(a_t \ \ | \ \ s_t \right) ">

# <img src="https://render.githubusercontent.com/render/math?math=2 \. \ \ \nabla_\theta J \left( \theta \right) \approx \Sigma_i \ \left( \Sigma_t \nabla_\theta \ log \  \pi_\theta \ \left(a_t^i \ \ | \ \ s_t^i \right) \Sigma_{t'=t}^T r \left(a_{t'}^i \ \ | \ \ s_{t'}^i \right) \right)">

# <img src="https://render.githubusercontent.com/render/math?math=3 \. \ \theta \leftarrow \theta \  \dotplus \alpha \nabla_\theta J \left( \theta \right) ">

<img width="647" alt="스크린샷 2022-05-23 오후 8 14 51" src="https://user-images.githubusercontent.com/24292848/169807378-9496b69c-bbec-4a45-ad6f-31b0215797ab.png">


## DQN (with static network)

$$ 1 \. save\ update \ network\ parameters\ to\ static\ network\ parameters: \phi ' \leftarrow \phi $$
$$ 2 \. collect dataset \left\{ \left( s_i, a_i, s_i ' , r_i \right) \right\} $$

<img width="647" alt="스크린샷 2022-05-23 오후 8 30 01" src="https://user-images.githubusercontent.com/24292848/169809875-e733ff29-249f-43e1-95bc-a9209d0d0ae7.png">

choose a2 and next state become s'. we calculate Q(s'a') from static network

<img width="603" alt="스크린샷 2022-05-23 오후 8 35 29" src="https://user-images.githubusercontent.com/24292848/169810706-73d3b59b-0db0-4176-8243-8b2fb24a0697.png">

update (nework_to_update) to reduce gap between Q(s,a2)(which is 2), reward(which is 5) + Q(s'a')(which is 5)

TRPO
DDPG
PPO
AC

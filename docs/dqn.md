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

![image](https://user-images.githubusercontent.com/24292848/181654349-dadf9857-6f49-457b-8af6-f19b2d2814b0.png)

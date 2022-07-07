## PG (with no baseline)

when we use replay memory(buffer), we have to use importance sampling and save importance weight in every trajectory. 
or we have to save trajectory which only used lastest policy.

$$ 1 \. \ \ sample \ (\tau^ i ) from \ \ \pi_\theta \ (a_t \ \ | \ \ s_t ) $$

$$ 2 \. \ \ \nabla_\theta J ( \theta ) \approx \Sigma_i \ ( \Sigma_t \nabla_\theta \ log \  \pi_\theta \ (a_t^i \ \ | \ \ s_t^i ) \Sigma_{t'=t}^T r (a_{t'}^i \ \ | \ \ s_{t'}^i ) ) $$

$$ 3 \. \ \theta \leftarrow \theta \  \dotplus \alpha \nabla_\theta J ( \theta ) $$


<img width="647" alt="스크린샷 2022-05-23 오후 8 14 51" src="https://user-images.githubusercontent.com/24292848/169807378-9496b69c-bbec-4a45-ad6f-31b0215797ab.png">

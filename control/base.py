class BasePolicy:
    """
    b_s batch_size
    ca capacity
    o_s observation space
    a_s action space
    h_s hidden space
    lr learning rate
    t_i training iteration
    """
    def __init__(self,
                 b_s,
                 ca,
                 o_s,
                 a_s,
                 h_s,
                 lr,
                 t_i
                 ):
        self.b_s = b_s
        self.ca = ca
        self.o_s = o_s
        self.a_s = a_s
        self.h_s = h_s
        self.lr = lr
        self.t_i = t_i



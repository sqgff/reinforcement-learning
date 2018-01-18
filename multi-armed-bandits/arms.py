import numpy as np
import matplotlib.pyplot as plt

class AbstractArm(object):
    def __init__(self, mean, variance, random_state):
        """
        Args:
            mean: expectation of the arm
            variance: variance of the arm
            random_state (int): seed to make experiments reproducible
        """
        self.mean = mean
        self.variance = variance

        self.local_random = np.random.RandomState(random_state)

    def sample(self):
        pass

class ArmPieceConstant(AbstractArm):
    def __init__(self, mean, delta=0.2, fre=100, random_state=0):
        """
        Piece wise constant arm
        """
        self.count = 0
        self.delta = delta
        self.frequence = fre
        super(ArmPieceConstant, self).__init__(mean=mean,
                                              variance=None,
                                              random_state=random_state)
        
    def sample(self):
        res = 0
        if self.count <= self.frequence:
            res = self.mean - self.delta
        elif self.count <= 2 * self.frequence:
            res = self.mean
        else:
            res = self.mean +self.delta
        self.count = (self.count + 1) % (3 * self.frequence)
        return res
    
class ArmPieceIncrease(AbstractArm):
    def __init__(self, lower=0, upper=1, delta=0.1, prob=0.001, random_state=0):
        self.value = lower
        self.upper = upper
        self.delta = delta
        self.prob = prob
        super(ArmPieceIncrease, self).__init__(mean=None,
                                              variance=None,
                                              random_state=random_state)
        
    def sample(self):
        if self.local_random.rand(1) < self.prob:
            self.value += self.delta
            if self.value > self.upper: self.value = self.upper
        return self.value
    
class ArmPieceDecrease(AbstractArm):
    def __init__(self, lower=0, upper=1, delta=0.1, prob=0.001, random_state=0):
        self.value = upper
        self.lower = lower
        self.delta = delta
        self.prob = prob
        super(ArmPieceDecrease, self).__init__(mean=None,
                                              variance=None,
                                              random_state=random_state)

    def sample(self):
        if self.local_random.rand(1) < self.prob:
            self.value -= self.delta
            if self.value < self.lower: self.value = self.lower
        return self.value

class ArmBernoulli(AbstractArm):
    def __init__(self, p, random_state=0):
        """
        Bernoulli arm
        Args:
             p (float): mean parameter
             random_state (int): seed to make experiments reproducible
        """
        self.p = p
        super(ArmBernoulli, self).__init__(mean=p,
                                           variance=p * (1. - p),
                                           random_state=random_state)

    def sample(self):
        return int(self.local_random.rand(1) < self.p)


class ArmBeta(AbstractArm):
    def __init__(self, a, b, random_state=0):
        """
        arm having a Beta distribution
        Args:
             a (float): first parameter
             b (float): second parameter
             random_state (int): seed to make experiments reproducible
        """
        self.a = a
        self.b = b
        super(ArmBeta, self).__init__(mean=a / (a + b),
                                      variance=(a * b) / ((a + b) ** 2 * (a + b + 1)),
                                      random_state=random_state)

    def sample(self):
        return self.local_random.beta(self.a, self.b, 1)


class ArmExp(AbstractArm):
    # https://en.wikipedia.org/wiki/Truncated_distribution
    # https://en.wikipedia.org/wiki/Exponential_distribution
    # http://lagrange.math.siu.edu/Olive/ch4.pdf
    def __init__(self, L, B=1., random_state=0):
        """
        pdf =
        Args:
            L (float): parameter of the exponential distribution
            B (float): upper bound of the distribution (lower is 0)
            random_state (int): seed to make experiments reproducible
        """
        assert B > 0.
        self.L = L
        self.B = B
        v_m = (1. - np.exp(-B*L)*(1. + B*L)) / L
        super(ArmExp, self).__init__(mean=v_m / (1. - np.exp(-L * B)),
                                     variance=None,  # compute it yourself!
                                     random_state=random_state)

    def cdf(self, x):
        cdf = lambda y: 1. - np.exp(-self.L*y)
        truncated_cdf = (cdf(x) - cdf(0)) / (cdf(self.B) - cdf(0))
        return truncated_cdf

    def inv_cdf(self, q):
        assert 0<= q <= 1.
        v = - np.log(1. - (1. - np.exp(- self.L * self.B)) * q) / self.L
        return v

    def sample(self):
        # Inverse transform sampling
        # https://en.wikipedia.org/wiki/Inverse_transform_sampling
        q = self.local_random.random_sample(1)
        x = self.inv_cdf(q=q)
        return x

class ArmFinite(AbstractArm):
    def __init__(self, X, P, random_state=0):
        """
        Arm with finite support
        Args:
            X: support of the distribution
            P: associated probabilities
            random_state (int): seed to make experiments reproducible
        """
        self.X = X
        self.P = P
        mean = np.sum(X * P)
        super(ArmFinite, self).__init__(mean=mean,
                                        variance=np.sum(X ** 2 * P) - mean ** 2,
                                        random_state=random_state)

    def sample(self):
        i = self.local_random.choice(len(self.P), size=1, p=self.P)
        reward = self.X[i]
        return reward


def main():
    arm1 = ArmPieceConstant(mean=0.5, delta=0.2, fre=500, random_state=0)
    arm2 = ArmPieceIncrease(lower=0, upper=1, delta=0.1, prob=0.001, random_state=0)
    arm3 = ArmPieceDecrease(lower=0, upper=1, delta=0.1, prob=0.001, random_state=0)

    arms = [arm1, arm2, arm3]

    T = 1e4
    loss = []
    for i in range(int(T)):
        loss.append([arm.sample() for arm in arms])
    loss =np.array(loss)

    plt.figure()
    labels = ['piece constant recycle', 'piece constant increase', 'piece constant decrease']
    for k in range(len(arms)):
        plt.plot(loss[:, k], label=labels[k])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

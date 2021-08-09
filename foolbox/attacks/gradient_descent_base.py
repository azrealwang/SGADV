from typing import Union, Any, Optional, Callable, Tuple
from abc import ABC, abstractmethod
import eagerpy as ep

from ..devutils import flatten
from ..devutils import atleast_kd

from ..types import Bounds

from ..models.base import Model

from ..criteria import Misclassification

from ..distances import l1, l2, linf

from .base import FixedEpsilonAttack
from .base import T
from .base import get_criterion
from .base import raise_if_kwargs

from ..utils import cos_similarity_score, convergence

from collections import deque
import torch



class BaseGradientDescent(FixedEpsilonAttack, ABC):
    def __init__(
        self,
        *,
        rel_stepsize: float,
        abs_stepsize: Optional[float] = None,
        steps: int,
        random_start: bool,
        threshold: Optional[float] = None,
        loss_type: str = 'ST', #'ST', 'MT', 'MS', 'C-BCE'
        convergence_threshold: Optional[float] = None,
        device: str = 'cpu',
        threshold2: Optional[float] = None,
    ):
        self.rel_stepsize = rel_stepsize
        self.abs_stepsize = abs_stepsize
        self.steps = steps
        self.random_start = random_start
        self.threshold=threshold
        self.loss_type=loss_type
        self.convergence_threshold=convergence_threshold
        self.device=device
        self.threshold2=threshold2

    def get_loss_fn(
        self, 
        model: Model, 
        labels: ep.Tensor, 
        #labels0: ep.Tensor, 
        model2: Model,
        labels2: ep.Tensor, 
    ) -> Callable[[ep.Tensor], ep.Tensor]:
        # can be overridden by users
        def loss_fn(inputs: ep.Tensor) -> ep.Tensor:
            logits = model(inputs)
            similarity = cos_similarity_score(logits.raw,labels.raw)
            if self.loss_type=='ST':
                loss = similarity-1
            elif self.loss_type=='MT':
                similarity2 = cos_similarity_score(logits.raw,labels2.raw)
                loss = similarity+similarity2-2
            elif self.loss_type=='MS':
                logits2 = model2(inputs)
                similarity2 = cos_similarity_score(logits2.raw,labels2.raw)
                loss = (similarity-1)/(1-self.threshold)+(similarity2-1)/(1-self.threshold2)
            elif self.loss_type=='C-BCE':
                threshold = torch.Tensor([self.threshold]*self.totalsize).to(self.device)
                loss = torch.log(torch.minimum(similarity, threshold)/(self.threshold))
            else:
                print("unsupported loss type")
                exit()
            
            loss = ep.astensor(loss.sum())
            print(loss)
            
            return loss

        return loss_fn

    def value_and_grad(
        # can be overridden by users
        self,
        loss_fn: Callable[[ep.Tensor], ep.Tensor],
        x: ep.Tensor,
    ) -> Tuple[ep.Tensor, ep.Tensor]:
        return ep.value_and_grad(loss_fn, x)

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Misclassification, T],
        criterion2: Optional[Union[Misclassification, T]] = None,
        model2: Optional[Model] = None,
        *,
        epsilon: float,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        x0, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion, kwargs

        #if not isinstance(criterion_, Misclassification):
        #    raise ValueError("unsupported criterion")
        #labels0 = model(x0)
        
        labels = criterion_.labels
        self.totalsize = labels.shape[0]
        labels2 = None
        if criterion2 is not None:
            criterion_2 = get_criterion(criterion2)
            del criterion2
            labels2 = criterion_2.labels
            self.totalsize = self.totalsize*2
        loss_fn = self.get_loss_fn(model, labels, model2, labels2)
        
        if self.abs_stepsize is None:
            stepsize = self.rel_stepsize * epsilon
        else:
            stepsize = self.abs_stepsize

        x = x0

        if self.random_start:
            x = self.get_random_start(x0, epsilon)
            x = ep.clip(x, *model.bounds)
        else:
            x = x0

        loss_deq = deque(maxlen=11)
        for _ in range(self.steps):
            loss, gradients = self.value_and_grad(loss_fn, x)
            #if (self.convergence_threshold is not None) and (self.loss_type != 'C-BCE'):
            if self.convergence_threshold is not None:
                loss_deq.append(loss.raw.tolist())
                if convergence(loss_deq,self.convergence_threshold*self.totalsize):
                    break
            gradients = self.normalize(gradients, x=x, bounds=model.bounds)
            x = x + stepsize * gradients
            x = self.project(x, x0, epsilon)
            x = ep.clip(x, *model.bounds)
        
        return restore_type(x)

    @abstractmethod
    def get_random_start(self, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        ...

    @abstractmethod
    def normalize(
        self, gradients: ep.Tensor, *, x: ep.Tensor, bounds: Bounds
    ) -> ep.Tensor:
        ...

    @abstractmethod
    def project(self, x: ep.Tensor, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        ...


def clip_lp_norms(x: ep.Tensor, *, norm: float, p: float) -> ep.Tensor:
    assert 0 < p < ep.inf
    norms = flatten(x).norms.lp(p=p, axis=-1)
    norms = ep.maximum(norms, 1e-12)  # avoid divsion by zero
    factor = ep.minimum(1, norm / norms)  # clipping -> decreasing but not increasing
    factor = atleast_kd(factor, x.ndim)
    return x * factor


def normalize_lp_norms(x: ep.Tensor, *, p: float) -> ep.Tensor:
    assert 0 < p < ep.inf
    norms = flatten(x).norms.lp(p=p, axis=-1)
    norms = ep.maximum(norms, 1e-12)  # avoid divsion by zero
    factor = 1 / norms
    factor = atleast_kd(factor, x.ndim)
    return x * factor


def uniform_l1_n_balls(dummy: ep.Tensor, batch_size: int, n: int) -> ep.Tensor:
    # https://mathoverflow.net/a/9188
    u = ep.uniform(dummy, (batch_size, n))
    v = u.sort(axis=-1)
    vp = ep.concatenate([ep.zeros(v, (batch_size, 1)), v[:, : n - 1]], axis=-1)
    assert v.shape == vp.shape
    x = v - vp
    sign = ep.uniform(dummy, (batch_size, n), low=-1.0, high=1.0).sign()
    return sign * x


def uniform_l2_n_spheres(dummy: ep.Tensor, batch_size: int, n: int) -> ep.Tensor:
    x = ep.normal(dummy, (batch_size, n + 1))
    r = x.norms.l2(axis=-1, keepdims=True)
    s = x / r
    return s


def uniform_l2_n_balls(dummy: ep.Tensor, batch_size: int, n: int) -> ep.Tensor:
    """Sampling from the n-ball

    Implementation of the algorithm proposed by Voelker et al. [#Voel17]_

    References:
        .. [#Voel17] Voelker et al., 2017, Efficiently sampling vectors and coordinates
            from the n-sphere and n-ball
            http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf
    """
    s = uniform_l2_n_spheres(dummy, batch_size, n + 1)
    b = s[:, :n]
    return b


class L1BaseGradientDescent(BaseGradientDescent):
    distance = l1

    def get_random_start(self, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        batch_size, n = flatten(x0).shape
        r = uniform_l1_n_balls(x0, batch_size, n).reshape(x0.shape)
        return x0 + epsilon * r

    def normalize(
        self, gradients: ep.Tensor, *, x: ep.Tensor, bounds: Bounds
    ) -> ep.Tensor:
        return normalize_lp_norms(gradients, p=1)

    def project(self, x: ep.Tensor, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        return x0 + clip_lp_norms(x - x0, norm=epsilon, p=1)


class L2BaseGradientDescent(BaseGradientDescent):
    distance = l2

    def get_random_start(self, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        batch_size, n = flatten(x0).shape
        r = uniform_l2_n_balls(x0, batch_size, n).reshape(x0.shape)
        return x0 + epsilon * r

    def normalize(
        self, gradients: ep.Tensor, *, x: ep.Tensor, bounds: Bounds
    ) -> ep.Tensor:
        return normalize_lp_norms(gradients, p=2)

    def project(self, x: ep.Tensor, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        return x0 + clip_lp_norms(x - x0, norm=epsilon, p=2)


class LinfBaseGradientDescent(BaseGradientDescent):
    distance = linf

    def get_random_start(self, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        return x0 + ep.uniform(x0, x0.shape, -epsilon, epsilon)

    def normalize(
        self, gradients: ep.Tensor, *, x: ep.Tensor, bounds: Bounds
    ) -> ep.Tensor:
        return gradients.sign()

    def project(self, x: ep.Tensor, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        return x0 + ep.clip(x - x0, -epsilon, epsilon)

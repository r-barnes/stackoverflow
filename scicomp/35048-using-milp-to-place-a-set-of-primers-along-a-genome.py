#!/usr/bin/env python3

from collections import deque
from functools import lru_cache
import copy
import random



def sliding_window_sum(a, size):
  assert size>0
  out     = []
  the_sum = 0
  q       = deque()
  for i in a:
    if len(q)==size:
      the_sum -= q[0]
      q.popleft()
    q.append(i)
    the_sum += i
    if len(q)==size:
      out.append(the_sum)
  return out



class Scoreifier:
  def __init__(
    self,
    v,               #Array of mutations
    lb_u:int   = 18, #Lower bound on inter-primer spacing
    ub_u:int   = 60, #Upper bound on inter-primer spacing
    lb_p:int   = 18, #Lower bound on primer length
    ub_p:int   = 24, #Upper bound on primer length
    pcount:int = 8   #Number of primers
  ):
    #Problem attributes
    self.v      = v 
    self.lb_u   = lb_u
    self.ub_u   = ub_u
    self.lb_p   = lb_p
    self.ub_p   = ub_p
    self.pcount = pcount
    #Cache some handy information for later (pulls a factor len(p) out of the
    #time complexity). Code is simplified at low cost of additional space by
    #calculating subarray sums we won't use.
    self.sub_sums = [[]] + [sliding_window_sum(v, i) for i in range(1, ub_p+1)]

  @staticmethod
  def _get_best(current_best, ret):
    if current_best is None:
      current_best = copy.deepcopy(ret)
    elif ret["score"]<current_best["score"]:
      current_best = copy.deepcopy(ret)
    elif ret["score"]==current_best["score"] and ret["cum_len"]>current_best["cum_len"]:
      current_best = copy.deepcopy(ret)

    return current_best

  @lru_cache(maxsize=None)
  def _find_best_helper(
    self,
    p,        #Primer we're currently considering
    start,    #Starting position for this primer
    plen      #Length of this primer
  ):
    #Don't consider primer location-length combinations that put us outside the
    #dataset
    if start>=len(self.sub_sums[plen]):
      return {
        "score":     float('inf'),
        "cum_len":   -float('inf'),
        "lengths":   [],
        "positions": []
      }
    elif p==self.pcount-1:
      return {
        "score":     self.sub_sums[plen][start],
        "cum_len":   plen,
        "lengths":   [plen],
        "positions": [start]
      }

    #Otherwise, find the best arrangement starting from the current location
    current_best = None
    for next_start in range(start+self.lb_u, start+self.ub_u+1):
      for next_plen in range(self.lb_p, self.ub_p+1):
        ret = self._find_best_helper(p=p+1, start=next_start, plen=next_plen)
        current_best = self._get_best(current_best, ret)

    current_best["score"]   += self.sub_sums[plen][start]
    current_best["cum_len"] += plen
    current_best["lengths"].append(plen)
    current_best["positions"].append(start)

    return current_best

  def find_best(self):
    #Consider all possible starting locations
    current_best = None
    for start in range(len(v)):
      print(f"Start: {start}")
      for plen in range(self.lb_p, self.ub_p+1):
        ret = self._find_best_helper(p=0, start=start, plen=plen)
        current_best = self._get_best(current_best, ret)

    return current_best        

G = 30_000
v = random.choices(population=[0,1], weights=[0.75, 0.25], k=G)

ret = Scoreifier(v=v).find_best()
print(ret)
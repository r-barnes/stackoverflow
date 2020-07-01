#include <boost/container_hash/extensions.hpp>

#include <cassert>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <vector>
#include <utility>
#include <unordered_map>

typedef std::vector<int> ivec;



struct Score {
  double score   = std::numeric_limits<double>::infinity();
  double cum_len = -std::numeric_limits<double>::infinity();
  ivec lengths;
  ivec positions;
  bool operator<(const Score &o) const {
    if(score<o.score)
      return true;
    else if(score==o.score && cum_len>o.cum_len)
      return true;
    else
      return false;
  }
};



typedef std::tuple<int,int,int> find_best_arg_type;

struct FBAThash {
  std::size_t operator()(const find_best_arg_type &key) const {
    return boost::hash_value(key);
  }
};

using FBATmap = std::unordered_map<find_best_arg_type, Score, FBAThash>;



template<class T>
std::vector<T> sliding_window_sum(const std::vector<T> &v, const int size){
  assert(size>0);
  std::vector<T> out;
  T the_sum = 0;
  std::deque<T> q;
  for(const auto &x: v){
    if(q.size()==size){
      the_sum -= q.front();
      q.pop_front();
    }
    q.push_back(x);
    the_sum += x;
    if(q.size()==size)
      out.push_back(the_sum);
  }  
  return out;
}



class Scoreifier {
 public:
  ivec v;
  const int lb_u;
  const int ub_u;
  const int lb_p;
  const int ub_p;
  const int pcount;

  Scoreifier(const ivec &v, int lb_u, int ub_u, int lb_p, int ub_p, int pcount):
    v(v), lb_u(lb_u), ub_u(ub_u), lb_p(lb_p), ub_p(ub_p), pcount(pcount)
  {
    //Cache some handy information for later (pulls a factor len(p) out of the
    //time complexity). Code is simplified at low cost of additional space by
    //calculating subarray sums we won't use.
    sub_sums.emplace_back(); //Empty array for 0
    for(int i=1;i<ub_p+1;i++)
      sub_sums.push_back(sliding_window_sum(v, i));
  }

  Score find_best(){
    //Consider all possible starting locations
    Score current_best;
    for(int start=0;start<v.size();start++){
      std::cout<<"Start: "<<start<<"\n";
      for(int plen=lb_p;plen<ub_p+1;plen++)
        current_best = std::min(current_best,find_best_helper(0, start, plen));
    }
    return current_best;
  }

 private:
  FBATmap visited;
  
  std::vector<ivec> sub_sums;

  Score find_best_helper(
    const int p,     //Primer we're currently considering
    const int start, //Starting position for this primer
    const int plen   //Length of this primer
  ){
    //Don't repeat if we've already solved this problem
    const auto key = find_best_arg_type(p,start,plen);
    if(visited.count(key)!=0)
      return visited.at(key);

    //Don't consider primer location-length combinations that put us outside the
    //dataset
    if(start>=sub_sums.at(plen).size())
      return {};
    else if(p==pcount-1)
      return {(double)sub_sums.at(plen).at(start), (double)plen, {plen}, {start}};

    //Otherwise, find the best arrangement starting from the current location
    Score current_best;
    for(int next_start=start+lb_u; next_start<start+ub_u+1; next_start++)
    for(int next_plen=lb_p; next_plen<ub_p+1; next_plen++)
      current_best = std::min(current_best, find_best_helper(p+1, next_start, next_plen));

    current_best.score   += sub_sums[plen][start];
    current_best.cum_len += plen;
    current_best.lengths.push_back(plen);
    current_best.positions.push_back(start);

    visited[key] = current_best;

    return current_best;
  }
};



int main(){
  const int G=30'000;
  
  ivec v;
  for(int i=0;i<G;i++){
    v.push_back(rand()%100<25);
  }
  
  const auto sc = Scoreifier(v, 18, 60, 18, 24, 8).find_best();

  std::cout<<"best_score      = "<<sc.score<<std::endl;
  std::cout<<"best_cum_length = "<<sc.cum_len<<std::endl;
  
  std::cout<<"best_lengths    = ";
  for(const auto &x: sc.lengths)
    std::cout<<x<<" ";
  std::cout<<std::endl;
  
  std::cout<<"best_positions  = ";
  for(const auto &x: sc.positions)
    std::cout<<x<<" ";
  std::cout<<std::endl;

  return 0;
}

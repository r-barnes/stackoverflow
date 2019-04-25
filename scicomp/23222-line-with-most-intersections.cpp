//Compile with: g++ -g 23222-line-with-most-intersections.cpp -lCGAL -lgmp -lmpfr

#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Arr_segment_traits_2.h>
#include <CGAL/Surface_sweep_2.h>
#include <CGAL/Surface_sweep_2_algorithms.h>
//#include <CGAL/Sweep_line_2.h>
#include <CGAL/Surface_sweep_2/Default_visitor.h>
#include <CGAL/Surface_sweep_2/Surface_sweep_2_utils.h>
#include <list>
#include <vector>

typedef CGAL::Exact_predicates_exact_constructions_kernel       Kernel;
typedef Kernel::Point_2                                         Point_2;
typedef CGAL::Arr_segment_traits_2<Kernel>                      Traits_2;
typedef Traits_2::Curve_2                                       Segment_2;


namespace CGAL {
namespace Surface_sweep_2 {

template <typename GeometryTraits_2, typename OutputIterator,
          typename Allocator_ = CGAL_ALLOCATOR(int)>
class IntersectionCounter :
  public Default_visitor<IntersectionCounter<GeometryTraits_2,
                                                      OutputIterator,
                                                     Allocator_>,
                         GeometryTraits_2, Allocator_>
{
public:
  typedef GeometryTraits_2                              Geometry_traits_2;
  typedef OutputIterator                                Output_iterator;
  typedef Allocator_                                    Allocator;

private:
  typedef Geometry_traits_2                             Gt2;
  typedef IntersectionCounter<Gt2, Output_iterator, Allocator>
                                                        Self;
  typedef Default_visitor<Self, Gt2, Allocator>         Base;

public:
  typedef typename Base::Event                          Event;
  typedef typename Base::Subcurve                       Subcurve;

  typedef typename Subcurve::Status_line_iterator       Status_line_iterator;

  typedef typename Gt2::X_monotone_curve_2              X_monotone_curve_2;
  typedef typename Gt2::Point_2                         Point_2;

  typedef typename Base::Surface_sweep_2                Surface_sweep_2;

protected:
  Output_iterator m_out;                 // The output points.

public:
  IntersectionCounter(Output_iterator out) :
    m_out(out)
  {}

  template <typename CurveIterator>
  void sweep(CurveIterator begin, CurveIterator end)
  {
    std::vector<X_monotone_curve_2> curves_vec;
    std::vector<Point_2> points_vec;

    curves_vec.reserve(std::distance(begin,end));
    make_x_monotone(begin, end,
                    std::back_inserter(curves_vec),
                    std::back_inserter(points_vec),
                    this->traits());

    //Original curves get converted into x-monotone curves here, but, since they
    //are segments, their ordering and data appears to be unaltered
    std::cout<<"x-monotone curves\n";
    for(auto &x: curves_vec)
      std::cout<<x<<" "<<(&x)<<std::endl;
    std::cout<<"x-monotone points\n";
    for(auto &x: points_vec)
      std::cout<<x<<std::endl;

    //Perform the sweep
    Surface_sweep_2* sl = this->surface_sweep();
    sl->sweep(curves_vec.begin(), curves_vec.end(),
              points_vec.begin(), points_vec.end());
  }

  bool after_handle_event(Event* event,
                          Status_line_iterator /* iter */,
                          bool /* flag */)
  {
    //TODO: Magic should happen here
    if ((
         event->is_intersection() ||
         event->is_weak_intersection()) && event->is_closed())
    {
      *m_out = event->point();
      ++m_out;
    }
    return true;
  }

  Output_iterator output_iterator() { return m_out; }
};

} // namespace Surface_sweep_2

namespace Ss2 = Surface_sweep_2;



template <typename CurveInputIterator, typename OutputIterator, typename Traits>
OutputIterator CountIntersections(
  CurveInputIterator curves_begin,
  CurveInputIterator curves_end,
  OutputIterator points,
  Traits &tr
){
  // Define the surface-sweep types:
  typedef Ss2::IntersectionCounter<Traits, OutputIterator> Visitor;
  typedef Ss2::Surface_sweep_2<Visitor>                    Surface_sweep;

  // Perform the sweep and obtain the intersection points.
  Visitor visitor(points);
  Surface_sweep surface_sweep(&tr, &visitor);
  visitor.sweep(curves_begin, curves_end);

  return visitor.output_iterator();
}



template <typename CurveInputIterator, typename OutputIterator>
OutputIterator CountIntersections(
  CurveInputIterator curves_begin,
  CurveInputIterator curves_end,
  OutputIterator points
){
  typedef typename std::iterator_traits<CurveInputIterator>::value_type  Curve;
  typename Default_arr_traits<Curve>::Traits   traits;
  return CountIntersections(curves_begin, curves_end, points, traits);
}


} // namespace CGAL



int main(){
  //Points as extracted from https://scicomp.stackexchange.com/q/23222/17088
  const std::vector<Point_2> pts = {
    Point_2( 57,931),
    Point_2(447,699),
    Point_2(899,748),
    Point_2(863,137),
    Point_2(530, 67),
    Point_2(142,282)
  };

  //Points are fully connected
  std::vector<Segment_2> segments;
  for(int i=0;  i<pts.size();i++)
  for(int j=i+1;j<pts.size();j++){
    segments.emplace_back(pts[i],pts[j]);
    std::cout<<pts[i]<<"\n"<<pts[j]<<"\n\n";
  }

  // Compute all intersection points.
  std::list<Point_2> ipts;
  CGAL::CountIntersections(segments.begin(), segments.end(), std::back_inserter(ipts));

  for(const auto &x: segments)
    std::cout<<(&x)<<std::endl;

  // Print the result.
  std::cout << "Found " << ipts.size() << " intersection points: " << std::endl;
  std::copy(ipts.begin(), ipts.end(),
            std::ostream_iterator<Point_2>(std::cout, "\n"));
  return 0;
}
//Finds the cropped Voronoi diagram of a set of points and saves it as WKT
//Compile with: g++ -O3 main.cpp -o power_diagramer.exe -Wall -lCGAL -lgmp
//Author: Richard Barnes (rbarnes.org)
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Regular_triangulation_filtered_traits_2.h>
#include <CGAL/Regular_triangulation_adaptation_traits_2.h>
#include <CGAL/Regular_triangulation_adaptation_policies_2.h>
#include <CGAL/Regular_triangulation_2.h>
#include <CGAL/Voronoi_diagram_2.h>
#include <CGAL/Boolean_set_operations_2.h>
#include <CGAL/bounding_box.h>
#include <CGAL/Polygon_2.h>
#include <iostream>
#include <cstdint>
#include <string>
#include <memory>

typedef CGAL::Exact_predicates_exact_constructions_kernel K;
typedef CGAL::Regular_triangulation_filtered_traits_2<K>  Traits;

typedef CGAL::Regular_triangulation_2<Traits> RT2;
typedef CGAL::Regular_triangulation_adaptation_traits_2<RT2>         AT;
typedef CGAL::Regular_triangulation_degeneracy_removal_policy_2<RT2> DRP;
typedef CGAL::Voronoi_diagram_2<RT2, AT, DRP> VD;

int main(int argc, char **argv){
  if(argc!=2){
    std::cerr<<"Synax: "<<argv[0]<<" <FILENAME>"<<std::endl;
    std::cerr<<"<FILENAME> may be a file or '-'. The latter reads from stdin."<<std::endl;
    std::cerr<<"File has the format:"<<std::endl;
    std::cerr<<"<RAY_LENGTH>"        <<std::endl;
    std::cerr<<"<CROP/NOCROP>"       <<std::endl;
    std::cerr<<"<X> <Y> <WEIGHT>"    <<std::endl;
    std::cerr<<"<X> <Y> <WEIGHT>"    <<std::endl;
    std::cerr<<"..."                 <<std::endl;
    std::cerr                        <<std::endl;
    std::cerr<<"'RAY_LENGTH' is a multiplier that extends infinite rays"<<std::endl;
    std::cerr<<"'CROP' will crop the power diagram to the bounding box of the input points"<<std::endl;
    return -1;
  }

  std::string filename = argv[1];

  //Output formatting
  std::cout.precision(4);          //Number of digits of decimal precision
  std::cout.setf(std::ios::fixed); //Don't use scientific notation

  //Used to convert otherwise infinite rays into looooong line segments by
  //multiplying the components of the direction vector of a ray by this value.
  int RAY_LENGTH = 1000;

  //Create a pointer to the correct input stream
  std::istream *fin;
  if(filename=="-")
    fin = &std::cin;
  else
    fin = new std::ifstream(filename);

  std::string crop_string;
  bool do_crop = false;

  (*fin)>>RAY_LENGTH;
  (*fin)>>crop_string;
  if(crop_string=="CROP")
    do_crop = true;
  else if(crop_string=="NOCROP")
    do_crop = false;
  else {
    std::cerr<<"Crop value must be 'CROP' or 'NOCROP'!"<<std::endl;
    return -1;
  }

  //Read in points from the command line
  RT2::Weighted_point wp;
  std::vector<RT2::Weighted_point> wpoints;
  while((*fin)>>wp)
    wpoints.push_back(wp);

  //Clean up the stream pointer
  if(filename!="-")
    delete fin;

  //Create a Regular Triangulation from the points
  RT2 rt(wpoints.begin(), wpoints.end());
  rt.is_valid();

  //Wrap the triangulation with a Voronoi diagram adaptor. This is necessary to
  //get the Voronoi faces.
  VD vd(rt);

  //CGAL often returns objects that are either segments or rays. This converts
  //these objects into segments. If the object would have resolved into a ray,
  //that ray is intersected with the bounding box defined above and returned as
  //a segment.
  const auto ConvertToSeg = [&](const CGAL::Object seg_obj, bool outgoing) -> K::Segment_2 {
    //One of these will succeed and one will have a NULL pointer
    const K::Segment_2 *dseg = CGAL::object_cast<K::Segment_2>(&seg_obj);
    const K::Ray_2     *dray = CGAL::object_cast<K::Ray_2>(&seg_obj);
    if (dseg) { //Okay, we have a segment
      return *dseg;
    } else {    //Must be a ray
      const auto &source = dray->source();
      const auto dsx     = source.x();
      const auto dsy     = source.y();
      const auto &dir    = dray->direction();
      const auto tpoint  = K::Point_2(dsx+RAY_LENGTH*dir.dx(),dsy+RAY_LENGTH*dir.dy());
      if(outgoing)
        return K::Segment_2(
          dray->source(),
          tpoint
        );
      else
        return K::Segment_2(
          tpoint,
          dray->source()
        );
    }
  };

  //Loop over the faces of the Voronoi diagram in some arbitrary order
  for(VD::Face_iterator fit = vd.faces_begin(); fit!=vd.faces_end();++fit){
    CGAL::Polygon_2<K> pgon;

    //Edge circulators traverse endlessly around a face. Make a note of the
    //starting point so we know when to quit.
    VD::Face::Ccb_halfedge_circulator ec_start = fit->ccb();

    //Find a bounded edge to start on
    for(;ec_start->is_unbounded();ec_start++){}

    //Current location of the edge circulator
    VD::Face::Ccb_halfedge_circulator ec = ec_start;

    //In WKT format each polygon must begin and end with the same point
    K::Point_2 first_point;

    do {
      //A half edge circulator representing a ray doesn't carry direction
      //information. To get it, we take the dual of the dual of the half-edge.
      //The dual of a half-edge circulator is the edge of a Delaunay triangle.
      //The dual of the edge of Delaunay triangle is either a segment or a ray.
      // const CGAL::Object seg_dual = rt.dual(ec->dual());
      const CGAL::Object seg_dual = vd.dual().dual(ec->dual());

      //Convert the segment/ray into a segment
      const auto this_seg = ConvertToSeg(seg_dual, ec->has_target());

      pgon.push_back(this_seg.source());
      if(ec==ec_start)
        first_point = this_seg.source();

      //If the segment has no target, it's a ray. This means that the next
      //segment will also be a ray. We need to connect those two rays with a
      //segment. The following accomplishes this.
      if(!ec->has_target()){
        const CGAL::Object nseg_dual = vd.dual().dual(ec->next()->dual());
        const auto next_seg = ConvertToSeg(nseg_dual, ec->next()->has_target());
        pgon.push_back(next_seg.target());
      }
    } while ( ++ec != ec_start ); //Loop until we get back to the beginning

    if(do_crop){
      //Find the bounding box of the points. This will be used to crop the Voronoi
      //diagram later.
      const K::Iso_rectangle_2 bbox = CGAL::bounding_box(wpoints.begin(), wpoints.end());

      //In order to crop the Voronoi diagram, we need to convert the bounding box
      //into a polygon. You'd think there'd be an easy way to do this. But there
      //isn't (or I haven't found it).
      CGAL::Polygon_2<K> bpoly;
      bpoly.push_back(K::Point_2(bbox.xmin(),bbox.ymin()));
      bpoly.push_back(K::Point_2(bbox.xmax(),bbox.ymin()));
      bpoly.push_back(K::Point_2(bbox.xmax(),bbox.ymax()));
      bpoly.push_back(K::Point_2(bbox.xmin(),bbox.ymax()));

      //Perform the intersection. Since CGAL is very general, it believes the
      //result might be multiple polygons with holes.
      std::list<CGAL::Polygon_with_holes_2<K>> isect;
      CGAL::intersection(pgon, bpoly, std::back_inserter(isect));

      //But we know better. The intersection of a convex polygon and a box is
      //always a single polygon without holes. Let's assert this.
      assert(isect.size()==1);

      //And recover the polygon of interest
      auto &poly_w_holes = isect.front();
      pgon               = poly_w_holes.outer_boundary();
    }

    //Print the polygon as a WKT polygon
    std::cout<<"POLYGON ((";
    for(auto v=pgon.vertices_begin();v!=pgon.vertices_end();v++)
      std::cout<<v->x()<<" "<<v->y()<<", ";
    std::cout<<pgon.vertices_begin()->x()<<" "<<pgon.vertices_begin()->y()<<"))\n";
  }

  return 0;
}
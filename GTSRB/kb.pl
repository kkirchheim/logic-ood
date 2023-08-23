% knowledge base for GTSRB

trafficSign(pedestrian_crossing).
trafficSign(speed_limit_20).
trafficSign(speed_limit_30).
trafficSign(no_overtaking_trucks).
trafficSign(priority_at_next_intersection).
trafficSign(priority_road).
trafficSign(give_way).
trafficSign(stop).
trafficSign(no_traffic_both_ways).
trafficSign(no_trucks).
trafficSign(no_entry).
trafficSign(danger).
trafficSign(bend_left).
trafficSign(speed_limit_50).
trafficSign(bend_right).
trafficSign(bend).
trafficSign(uneven_road).
trafficSign(slippery_road).
trafficSign(road_narrows).
trafficSign(construction).
trafficSign(traffic_signal).
trafficSign(school_crossing).
trafficSign(cycles_crossing).
trafficSign(speed_limit_60).
trafficSign(snow).
trafficSign(animals).
trafficSign(restriction_ends).
trafficSign(go_right).
trafficSign(go_left).
trafficSign(go_straight).
trafficSign(go_right_or_straight).
trafficSign(go_left_or_straight).
trafficSign(keep_right).
trafficSign(keep_left).
trafficSign(speed_limit_70).
trafficSign(roundabout).
trafficSign(restriction_ends_overtaking).
trafficSign(restriction_ends_overtaking_trucks).
trafficSign(speed_limit_80).
trafficSign(restriction_ends_80).
trafficSign(speed_limit_100).
trafficSign(speed_limit_120).
trafficSign(no_overtaking).


shape(pedestrian_crossing, triangle).
shape(speed_limit_20, circle).
shape(speed_limit_30, circle).
shape(no_overtaking_trucks, circle).
shape(priority_at_next_intersection, triangle).
shape(priority_road, square).
shape(give_way, inverse_triange).
shape(stop, octagon).
shape(no_traffic_both_ways, circle).
shape(no_trucks, circle).
shape(no_entry, circle).
shape(danger, triangle).
shape(bend_left, triangle).
shape(speed_limit_50, circle).
shape(bend_right, triangle).
shape(bend, triangle).
shape(uneven_road, triangle).
shape(slippery_road, triangle).
shape(road_narrows, triangle).
shape(construction, triangle).
shape(traffic_signal, triangle).
shape(school_crossing, triangle).
shape(cycles_crossing, triangle).
shape(speed_limit_60, circle).
shape(snow, triangle).
shape(animals, triangle).
shape(restriction_ends, circle).
shape(go_right, circle).
shape(go_left, circle).
shape(go_straight, circle).
shape(go_right_or_straight, circle).
shape(go_left_or_straight, circle).
shape(keep_right, circle).
shape(keep_left, circle).
shape(speed_limit_70, circle).
shape(roundabout, circle).
shape(restriction_ends_overtaking, circle).
shape(restriction_ends_overtaking_trucks, circle).
shape(speed_limit_80, circle).
shape(restriction_ends_80, circle).
shape(speed_limit_100, circle).
shape(speed_limit_120, circle).
shape(no_overtaking, circle).



color(pedestrian_crossing, red).
color(speed_limit_20, red).
color(speed_limit_30, red).
color(no_overtaking_trucks, red).
color(priority_at_next_intersection, red).
color(priority_road, yellow).
color(give_way, red).
color(stop, red).
color(no_traffic_both_ways, red).
color(no_trucks, red).
color(no_entry, red).
color(danger, red).
color(bend_left, red).
color(speed_limit_50, red).
color(bend_right, red).
color(bend, red).
color(uneven_road, red).
color(slippery_road, red).
color(road_narrows, red).
color(construction, red).
color(traffic_signal, red).
color(school_crossing, red).
color(cycles_crossing, red).
color(speed_limit_60, red).
color(snow, red).
color(animals, red).
color(restriction_ends, white).
color(go_right, blue).
color(go_left, blue).
color(go_straight, blue).
color(go_right_or_straight, blue).
color(go_left_or_straight, blue).
color(keep_right, blue).
color(keep_left, blue).
color(speed_limit_70, red).
color(roundabout, blue).
color(restriction_ends_overtaking, white).
color(restriction_ends_overtaking_trucks, white).
color(speed_limit_80, red).
color(restriction_ends_80, white).
color(speed_limit_100, red).
color(speed_limit_120, red).
color(no_overtaking, red).

rotation(Sign, 0) :- trafficSign(Sign).

sign(Sign, true) :- trafficSign(Sign).

is_sat(Sign, Color, Shape, Rotation, IsSign) :-
    trafficSign(Sign),
    color(Sign, Color),
    shape(Sign, Shape),
    rotation(Sign, Rotation),
    sign(Sign, IsSign).


% example:
% is_sat(stop, red, octagon, 0, true).

function desired = command_point(t)

height = 0;

desired.x = [1, 0, -height]';
desired.v = [0, 0, 0]';
desired.x_2dot = [0, 0, 0]';
desired.x_3dot = [0, 0, 0]';
desired.x_4dot = [0, 0, 0]';


desired.b1 = [1, 0, 0]';
desired.b1_dot = [0, 0, 0]';
desired.b1_2dot = [0, 0, 0]';

end
function generate_case2_rotation_candidates

phi = (1+sqrt(5))/2;

vertices = [
    1, 1, 1;
    1, 1, -1;
    1, -1, 1;
    1, -1, -1;
    -1, 1, 1;
    -1, 1, -1;
    -1, -1, 1;
    -1, -1, -1;
    
    0, 1/phi, phi;
    0, 1/phi, -phi;
    0, -1/phi, phi;
    0, -1/phi, -phi;
    
    phi, 0, 1/phi;
    phi, 0, -1/phi;
    -phi, 0, 1/phi;
    -phi, 0, -1/phi;
    
    1/phi, phi, 0;
    -1/phi, phi, 0;
    1/phi, -phi, 0;
    -1/phi, -phi, 0;];

%% edges
edges = zeros(15, 2);
len = norm(vertices(1,:)-vertices(9,:));
idx = 1;
for i = 1:size(vertices,1)-1
  for j = i+1:size(vertices,1)
    if abs( norm(vertices(i,:) - vertices(j,:)) - len ) < 0.0001
      break_flg = false;
      for k = 1:idx-1
        if norm( vertices(i,:) + vertices(j,:) + vertices(edges(k,1),:) + vertices(edges(k,2),:)) < 0.0001
          break_flg = true;
          break
        end
      end
      if break_flg;
        break
      end
      edges(idx,1) = i;
      edges(idx,2) = j;
      idx = idx + 1;
    end
  end
end

% %% edges_all
% edges_all = zeros(30, 2);
% idx = 1;
% for i = 1:size(vertices,1)-1
%   for j = i+1:size(vertices,1)
%     if abs( norm(vertices(i,:) - vertices(j,:)) - len ) < 0.0001
%       edges_all(idx,1) = i;
%       edges_all(idx,2) = j;
%       idx = idx + 1;
%     end
%   end
% end

%% faces
faces(1,:) = [1 13 3 11 9];
faces(2,:) = [1 17 2 14 13];
faces(3,:) = [1 9 5 18 17];
faces(4,:) = [10 6 18 17 2];
faces(5,:) = [13 14 4 19 3];
faces(6,:) = [9 11 7 15 5];

%% 0. original
inds = 1:20;

%% 1. axis: vertex, angle: 2/3 * pi
idx = 2;
for x=1:size(vertices,1)
  vert_b = vertices(x,:);
  vert_b = vert_b ./ norm(vert_b);
  vertices_new = my_rotate( vert_b', 2 * pi / 3 ) * vertices';
  vertices_new = vertices_new';
  
  for i=1:size(vertices,1)
    for j=1:size(vertices,1)
      if sum(abs(vertices_new(i,:) - vertices(j,:))) < 0.0001
        inds(idx,i) = j;
      end
    end
  end
  idx = idx + 1;
end

%% 2. axis: middle point of edge, angle: pi
for x=1:size(edges,1)
  vert_b = vertices(edges(x,1),:) + vertices(edges(x,2),:);
  vert_b = vert_b ./ norm(vert_b);
  vertices_new = my_rotate( vert_b', pi ) * vertices';
  vertices_new = vertices_new';
  
  for i=1:size(vertices,1)
    for j=1:size(vertices,1)
      if sum(abs(vertices_new(i,:) - vertices(j,:))) < 0.0001
        inds(idx,i) = j;
      end
    end
  end
  idx = idx + 1;
end

%% 3. axis: center point of face, angle: (1, 2, 3, 4) * 2/5 pi
for x=1:size(faces,1)
  vert_b = vertices(faces(x,1),:) + vertices(faces(x,2),:) + vertices(faces(x,3),:) + vertices(faces(x,4),:) + vertices(faces(x,5),:) ;
  vert_b = vert_b ./ norm(vert_b);
  for y=1:4
    vertices_new = my_rotate( vert_b', y * 2 * pi / 5 ) * vertices';
    vertices_new = vertices_new';
    
    for i=1:size(vertices,1)
      for j=1:size(vertices,1)
        if sum(abs(vertices_new(i,:) - vertices(j,:))) < 0.0001
          inds(idx,i) = j;
        end
      end
    end
    idx = idx + 1;
  end
end

%% show rotation candidates
inds

end



function R = my_rotate(k,fi)
x = k(1);
y = k(2);
z = k(3);

R = zeros(3,3);

R(1,1) = cos(fi)+x^2*(1-cos(fi));
R(1,2) = x*y*(1-cos(fi))-z*sin(fi);
R(1,3) = x*z*(1-cos(fi))+y*sin(fi);

R(2,1) = y*x*(1-cos(fi))+z*sin(fi);
R(2,2) = cos(fi)+y^2*(1-cos(fi));
R(2,3) = y*z*(1-cos(fi))-x*sin(fi);

R(3,1) = z*x*(1-cos(fi))-y*sin(fi);
R(3,2) = z*y*(1-cos(fi))+x*sin(fi);
R(3,3) = cos(fi)+z^2*(1-cos(fi));
end
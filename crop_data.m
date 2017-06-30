clc
close all
v = VideoReader('converted.avi');

fig=figure;
for img = 1:75:v.NumberOfFrames; 

b = read(v, img); 
imshow(b);
[x,y]=getpts(fig)
for i=1:2:numel(x)-1
c=imcrop(b,[x(i) y(i) x(i+1)-x(i) (y(i+1)-y(i))]);
size(c)
filename=strcat(num2str(img),num2str(i,'%04i'),'.jpg');
imwrite(c,filename)

end

end

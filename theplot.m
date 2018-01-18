
fd=fopen('data\\outdata','rb');
cnt=69;
accelbias=[-755.40186823, -359.90878696, -874.79820164];
magbias=[-21.55728099, 104.87445398,  -70.38780774];
accelxM=fread(fd,cnt,'double');
accelyM=fread(fd,cnt,'double');
accelzM=fread(fd,cnt,'double');
magxM=fread(fd,cnt,'double');
magyM=fread(fd,cnt,'double');
magzM=fread(fd,cnt,'double');
fclose(fd);
axis equal;
figure(1)
h1=scatter3(accelxM,accelyM,accelzM,'.');
hold on
scatter3(0,0,0,'*')
scatter3(accelbias(1),accelbias(2),accelbias(3),'o')
%plot3([accelbias(1),0],[accelbias(2),0],[accelbias(3),0])
xlabel('accel\_x')
ylabel('accel\_y')
zlabel('accel\_z')
hold off
figure(2)
h2=scatter3(magxM,magyM,magzM,'.');
hold on
scatter3(magbias(1),magbias(2),magbias(3),'o')
scatter3(0,0,0,'*')
%plot3([magbias(1),0],[magbias(2),0],[magbias(3),0])
xlabel('mag\_x')
ylabel('mag\_y')
zlabel('mag\_z')
% scatter(accelxM,accelyM);
% figure(3)
% scatter(accelyM,accelzM);
% figure(4)
% scatter(accelxM,accelzM);
%scatter3(magxM,magyM,magzM)
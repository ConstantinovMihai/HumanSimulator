u_david_li_python = csvread('u_david_li.csv');
f_star_david_li_python = csvread('f_star_david_li.csv');

figure;
subplot(2,1,1);
hold on;
plot(t, u_sim, 'DisplayName', 'U sim');
plot(t, u_david_li_python, 'DisplayName', 'U david li python');
hold off;
title('Measured Response');
xlabel('Time (s)');
ylabel('Response');
legend show;
grid on;

subplot(2,1,2);
hold on;
plot(t, f_star , 'DisplayName', 'f star matlab');
plot(t, f_star_david_li_python, "DisplayName", 'F star david li python')
hold off;
xlabel('Time (s)');
ylabel('Difference in Response');
legend show;
grid on;
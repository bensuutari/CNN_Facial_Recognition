clear all;
close all;
fontsize=15;
linewidthval=2;
plotcolor='b';
plotcolor_nodropout='r';
dirname='/home/ben/Dropbox/python/facial_rec_tensorflow_convnet/results/';
websitedir='/home/ben/Dropbox/Ben_Suutari_Website/mysite/personal/static/personal/img/';

accuracy=csvread(strcat(dirname,'accuracy'));
loss=csvread(strcat(dirname,'loss'));
accuracy_nodropout=csvread(strcat(dirname,'accuracy_nodropout'));
loss_nodropout=csvread(strcat(dirname,'loss_nodropout'));


figure;plot(accuracy*100,plotcolor,'LineWidth',3);hold on;
set(gca,'LineWidth',linewidthval,'FontSize',fontsize);
box off;
axis square;
xlabel('Epoch #','FontSize',fontsize);
ylabel('Test Accuracy (%)','FontSize',fontsize);
ylim([0 100]);
saveas(gcf,strcat(dirname,'CNNaccuracy_graph'),'png');
saveas(gcf,strcat(websitedir,'CNNaccuracy_graph'),'png');
plot(accuracy_nodropout*100,plotcolor_nodropout,'LineWidth',3);
leg=legend('With Dropout','Without Dropout');
set(leg,'box','off','FontSize',fontsize,'Location','SouthEast');
saveas(gcf,strcat(dirname,'CNNaccuracy_graph_dropoutvsnodropout'),'png');
saveas(gcf,strcat(websitedir,'CNNaccuracy_graph_dropoutvsnodropout'),'png');



figure;plot(loss,plotcolor,'LineWidth',3);hold on;
set(gca,'LineWidth',linewidthval,'FontSize',fontsize);hold on;
box off;
axis square;
xlabel('Epoch #','FontSize',fontsize);
ylabel('Loss','FontSize',fontsize);
saveas(gcf,strcat(dirname,'CNNloss_graph'),'png');
saveas(gcf,strcat(websitedir,'CNNloss_graph'),'png');
plot(loss,plotcolor,'LineWidth',3);hold on;
saveas(gcf,strcat(dirname,'CNNloss_graph'),'png');
saveas(gcf,strcat(websitedir,'CNNloss_graph'),'png');
plot(loss_nodropout*100,plotcolor_nodropout,'LineWidth',3);
leg=legend('With Dropout','Without Dropout');
set(leg,'box','off','FontSize',fontsize);
saveas(gcf,strcat(dirname,'CNNloss_graph_dropoutvsnodropout'),'png');
saveas(gcf,strcat(websitedir,'CNNloss_graph_dropoutvsnodropout'),'png');

figure;hold on;
subplot(1,2,1);
plot(loss,plotcolor,'LineWidth',3);
set(gca,'LineWidth',linewidthval,'FontSize',fontsize);
box off;
axis square;
xlabel('Epoch #','FontSize',fontsize);
ylabel('Loss','FontSize',fontsize);
subplot(1,2,2);
plot(accuracy*100,plotcolor,'LineWidth',3);
set(gca,'LineWidth',linewidthval,'FontSize',fontsize);
box off;
axis square;
xlabel('Epoch #','FontSize',fontsize);
ylabel('Test Accuracy (%)','FontSize',fontsize);
ylim([0 100]);
saveas(gcf,strcat(dirname,'CNNloss_accuracy_graph'),'png');
saveas(gcf,strcat(websitedir,'CNNloss_accuracy_graph'),'png');




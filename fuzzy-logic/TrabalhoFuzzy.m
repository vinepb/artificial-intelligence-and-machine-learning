clear
clc
close all

%Dados da planta
beta = 0.4;
C1=pi*0.35^2;
C2=beta*C1;

Q_max = 0.0083; % 8,3 l/s

H1_max = 5.6;
H2_max = 4;

R2 = 4/Q_max;
R1 = 0.4*R2;

T = 0.1;   

%% plotagem dos gráficos

% LogicaFuzzy
fuzzyLevelSP =  FuzzyLogic.signals(1).values(:,1);
fuzzyLevel =  FuzzyLogic.signals(1).values(:,2);
fuzzyControlAction = FuzzyLogic.signals(2).values;
fuzzyErro = FuzzyLogic.signals(3).values;

fuzzyTime = 0:T:3000;
figure()
subplot(3,1,1)
plot(fuzzyTime,fuzzyLevelSP), xlabel('Tempo'), ylabel('Nível'), axis([0 max(fuzzyTime) 0 max([fuzzyLevel])*1.2]);
hold on
plot(fuzzyTime,fuzzyLevel,'r'), xlabel('Tempo'), legend('SP','PV');
subplot(3,1,2)
plot(fuzzyTime,fuzzyControlAction), xlabel('Tempo'), ylabel('Ação de Control'), axis([0 max(fuzzyTime) 0 max([fuzzyControlAction])*1.2]), legend('MV');;
subplot(3,1,3)
plot(fuzzyTime,fuzzyErro), xlabel('Tempo'), ylabel('Erro'), axis([0 max(fuzzyTime) 0 max([fuzzyErro])*1.2]),legend('Erro');

% Controle PI

PI_LevelSP =  PI_Control.signals(1).values(:,1);
PI_Level =  PI_Control.signals(1).values(:,2);
PI_ControlAction = PI_Control.signals(2).values;
PI_Erro = PI_Control.signals(3).values;

PI_Time = 0:T:3000;
figure()
subplot(3,1,1)
plot(PI_Time,PI_LevelSP), xlabel('Tempo'), ylabel('Nível'), axis([0 max(PI_Time) 0 max([PI_Level])*1.2]);
hold on
plot(PI_Time,PI_Level,'r'), xlabel('Tempo'), legend('SP','PV');
subplot(3,1,2)
plot(PI_Time,PI_ControlAction), xlabel('Tempo'), ylabel('Ação de Controle'), axis([0 max(PI_Time) 0 max([PI_ControlAction])*1.2]), legend('MV');;
subplot(3,1,3)
plot(PI_Time,PI_Erro), xlabel('Tempo'), ylabel('Erro'), axis([0 max(PI_Time) 0 max([PI_Erro])*1.2]),legend('Erro');
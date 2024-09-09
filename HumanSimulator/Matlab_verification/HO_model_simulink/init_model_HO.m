%% Author: Carmen Gomez Mena
%%% Date: March 2024 - Nov 2024
%%% MSc Thesis: A-HSC / C&S Track

%% File: Initiates the parameters for model_HO.slx

% Inputs:
%   - Simulation parameters
%   - Model definition (tau, dynamics, K_HO)
%   - Remnant gain (Kn)

% Outputs:
%   - Loads into workspace the model parameters (Hoe, H_stick, Hnms, tau's, CE...) 

%%% NOTE -> Make sure to select the correct K_inv_stick model. The full
%%% model might be improper in DI dynmamics

function [] = init_model_HO(simtime, dyn,tau_p_,K_HO_, K_n_)
    
    % Plotting stuff
    set(groot,'defaulttextinterpreter','latex');
    set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
    set(groot, 'defaultLegendInterpreter','latex');
    
    %load model
    load_system("model_HO.slx");

    % Set default stop time to 20 seconds
    set_param('model_HO','StopTime',num2str(simtime));
    set_param('model_HO',"FastRestart","on")
    %% Parameters for forcing functions
    
    load('fofu_tab.mat','fofu_tab'); % load forcing functions data
    fofu_tab = table2array(fofu_tab);
    % The forcing function table has 10 rows (one per sinusoidal) and 11
    % columns (t = target, d = disturbance):
        % 1: index
        % 2: k_t - integer that multiplies the base freq. 0.0524 rad/s
        % 3: A_t - Amplitude
        % 4: w_t - Frequency
        % 5,6,7: phit1/phi_t2/phit3 - phase shifts (3 realizations)
        % 8: k_d - integer that multiplies the base freq. 0.0524 rad/s
        % 9: A_d - Amplitude
        % 10: w_d - Frequency
        % 11: phi_d - phase
    
    fofu_nr = 1;  %3 possible phases for the forcing functions (select 1,2 or 3)
    % hardcode to 1, like Span

    %% Dynamics
    dynamics = dyn; % 1 = SI, 2 = DI
    
    %% Preview time 
    tau_p = tau_p_; % tau of preview for HO (time shift in forcing function)
    K_HO = K_HO_; 
    
    fprintf('### INITIALISING HO MODEL ###')
    fprintf('\nSimulation time: %.2f s', simtime)
    fprintf('\nControlled Element dynamics: %d', dynamics)
    fprintf('\nHO Preview time: %.2f s', tau_p)
    fprintf('\nKn : %.3f \n', K_n_)
    
    %% HO model parameters for SI dynamics
    if dynamics == 1
            % These are equal between McKenzie (Table 2.1) and Span for SI dynamics)
           
            K_f = 1; % / - far-view point response gain
            T_lf = 0.2; %s - far-view point lag constant
            
            Ke_HO = 1.25; % / - internal error response gain
            omega_nms = 10.5; %rad/s - nms break frequency
            zeta_nms = 0.35; % / - nms damping ratio
            tau_v = 0.26; %s - nms response time delay 
            
            % K_n = 0.08, 3.4, 0.223
            K_n = 0.03;
            w_bn = 1/0.23; % = 1/T_ln - remnant break freqnecy for SI (Span & McKenzie)
            
            %do the seed before calling every model, not when initializing
            % seed = num2str(randi([10000, 99999])); % seed
            % set_param('model_HO/Human Operator - van der El model  (far-view point only)/White Noise','seed',seed); % set seed
            % 
            % Sidestick model parameters
            K_stick = 10;  % [inches/rad] stick gain 0.44 [cm/deg] = 25.4 cm/rad
            k = 3.58;      % [Nm/rad] stifness
            b = 0.22;      % [Nms/rad] damping
            I = 0.01;      % [kg/m^2] inertia

            % K_stick = 0.44;  % [cm/deg]!!!! os u is in cm!
            
            % Inverse stick gain (simplified by McKenzie)
            % inv_stick = (k/K_stick);
                % Uncomment to use Span's K_inv
            inv_stick = tf([I b k], K_stick);

            % Controlled element of SI
            K_ce = 1.5;         % [-] controlled element gain
                
            %% Define TF for Hoe & Hce (different for SI or DI)
            
            H_oe = tf([0 Ke_HO], [0 1]); %equalizing TF
            H_ce = tf([0 K_ce], [1 0]); 
    
            H_nms = tf([omega_nms*omega_nms],[1 2*zeta_nms*omega_nms omega_nms*omega_nms]);
    
            H_star = H_oe*H_nms*inv_stick; %combined H_oe*H_nms (no time delay)
            
    end 
    
    if dynamics == 2 %% DI
            %McKenzie parameters with DI dynamics
            K_f = 1.07; % / - far-view point response gain
            T_lf = 0.95; %s - far-view point lag constant
            
            Ke_HO = 0.23; % / - internal error response gain
            T_Le = 1.76; % lag of response due to DI
    
            omega_nms = 10.4;
            %rad/s - nms break frequency
            zeta_nms = 0.24; % / - nms damping ratio
            tau_v = 0.33; %s - nms response time delay 
            
            %need ot doublecheck the remnant because McKenzie didn't do DI
            K_n = 0;
            w_bn = 1/0.23; % = 1/T_ln - remnant break freqnecy for DI (Span)
            

            % Sidestick model parameters
            K_stick = 10;  % [inches/rad] stick gain 0.44 [cm/deg] = 25.4 cm/rad??
            k = 3.58;      % [Nm/rad] stifness
            b = 0.22;      % [Nms/rad] damping
            I = 0.01;      % [kg/m^2] inertia
            
           
            % Inverse stick gain (simplified by McKenzie)
            inv_stick = (k/K_stick);
                
            % Controlled element of SI
            K_ce = 5;         % [-] controlled element gain
                
            %% Define TF for Hoe & Hce (different for SI or DI)
    
            H_oe = tf([Ke_HO*T_Le Ke_HO], [0 0 1]); %equalizing TF
            H_ce = tf([0 K_ce], [1 0 0]);
            H_nms = tf([omega_nms*omega_nms],[1 2*zeta_nms*omega_nms omega_nms*omega_nms]);
            H_star = (H_oe*H_nms*inv_stick); %combined H_oe*H_nms (no time delay)  
    end 

    %% Assign parameters so they can be used in simulink
    
        % FF definition
    assignin('base',"fofu_tab",fofu_tab);
    assignin('base','fofu_nr',fofu_nr);
        % HO parameters
    assignin('base',"tau_p",tau_p);
    assignin('base',"K_HO",K_HO);
        % far-ahead preview
    assignin('base',"K_f",K_f);
    assignin('base',"T_lf",T_lf);      
        % HO dynamics (H_star = Heo*Hnms)
    assignin('base',"H_star",H_star);
    % assignin('base',"Ke_HO",Ke_HO);
    assignin('base',"tau_v",tau_v);
        %nms
    % assignin('base',"zeta_nms",zeta_nms);
    % assignin('base',"omega_nms",omega_nms);
        %stick dynamics
    assignin('base',"K_stick",K_stick);
    assignin('base',"k",k);
    assignin('base',"b",b);
    assignin('base',"I",I);
    assignin('base',"inv_stick",inv_stick);
        % CE dynamics
    assignin('base',"H_ce",H_ce);
        %remnant
    assignin('base',"K_n",K_n*0);
    assignin('base',"w_bn",w_bn);
    % assignin('base',"seed",seed);

end
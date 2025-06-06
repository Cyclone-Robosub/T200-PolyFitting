% ForceFromPWMVoltage.m

primaryVoltage = 16;

function ForceFromPWMVoltagePlot()

    % Define PWM and Voltage ranges
    pwm_vals = linspace(1100, 1900, 100);
    volt_vals = linspace(8, 20, 100);
    [PWM, V] = meshgrid(pwm_vals, volt_vals);

    % Compute force
    F = ForceFromPWMVoltage(PWM, V);

    % Plot the surface
    figure;
    surf(PWM, V, F);
    xlabel('PWM (µs)');
    ylabel('Voltage (V)');
    zlabel('Force (Kg f)');
    title('Force as a Function of PWM and Voltage');
    colorbar;
    grid on;
   % shading interp;

end

function force = ForceFromPWMVoltage(PWM, V)
    
%Set Main Operating Voltage to Determine Zero Band Used
    primaryVoltage = 16;

% Clip PWM between 1100 and 1900
    PWM = min(max(PWM, 1100), 1900);

    % Offset PWM
    P = PWM - 1500;

    % Initialize force
    force = zeros(size(P));

   if primaryVoltage < 16

    % Lower region: P <= -40
    idx_lower = P <= -40;
    Pl = P(idx_lower);
    Vl = V(idx_lower);

    force(idx_lower) = ...
        0.18450859598294592 + ...
        -0.03877245224750089 * Vl + ...
        0.00437326839182978 * Vl.^2 + ...
        -0.00014159192815009454 * Vl.^3 + ...
        0.0047430745735151875 * Pl + ...
        -0.0011779727652119001 * Pl .* Vl + ...
        0.00012626694389350853 * Pl .* Vl.^2 + ...
        -3.6521719807495034e-06 * Pl .* Vl.^3 + ...
        -3.394361152608232e-05 * Pl.^2 + ...
        7.081135717273334e-06 * Pl.^2 .* Vl + ...
        -6.937006893095161e-07 * Pl.^2 .* Vl.^2 + ...
        1.7574133886452198e-08 * Pl.^2 .* Vl.^3 + ...
        -1.0125491849106606e-07 * Pl.^3 + ...
        2.3956832126051456e-08 * Pl.^3 .* Vl + ...
        -2.0262771883546578e-09 * Pl.^3 .* Vl.^2 + ...
        5.312516822050221e-11 * Pl.^3 .* Vl.^3;

    % Upper region: P >= 40
    idx_upper = P >= 40;
    Pu = P(idx_upper);
    Vu = V(idx_upper);

    force(idx_upper) = ...
        -2.2103591356566494 + ...
        0.5020761226661944 * Vu + ...
        -0.038894245021323104 * Vu.^2 + ...
        0.0009685515042413979 * Vu.^3 + ...
        0.04287528029632993 * Pu + ...
        -0.010018667907150527 * Pu .* Vu + ...
        0.0007918978226055346 * Pu .* Vu.^2 + ...
        -1.965296626711245e-05 * Pu .* Vu.^3 + ...
        -0.00015245579360787697 * Pu.^2 + ...
        3.5803548508328624e-05 * Pu.^2 .* Vu + ...
        -2.3976087919487086e-06 * Pu.^2 .* Vu.^2 + ...
        5.395703879569961e-08 * Pu.^2 .* Vu.^3 + ...
        1.3322303527138988e-07 * Pu.^3 + ...
        -2.9258256623955067e-08 * Pu.^3 .* Vu + ...
        1.7469073167777621e-09 * Pu.^3 .* Vu.^2 + ...
        -3.0349724016130765e-11 * Pu.^3 .* Vu.^3;
        
    else
    % For primaryVoltage >= 16
    % Lower region: P <= -28
    idx_lower = P <= -28;
    Pl = P(idx_lower);
    Vl = V(idx_lower);

    force(idx_lower) = ...
        0.08254487160555046 + ...
        -0.019480220675947152 * Vl + ...
        0.0028936369449359288 * Vl.^2 + ...
        -0.00010138307572005322 * Vl.^3 + ...
        0.003229898558180788 * Pl + ...
        -0.0008968265330310558 * Pl .* Vl + ...
        0.00010465994797731376 * Pl .* Vl.^2 + ...
        -3.055535261088971e-06 * Pl .* Vl.^3 + ...
        -4.055726784727783e-05 * Pl.^2 + ...
        8.295219187479552e-06 * Pl.^2 .* Vl + ...
        -7.871417820116173e-07 * Pl.^2 .* Vl.^2 + ...
        2.018189374507016e-08 * Pl.^2 .* Vl.^3 + ...
        -1.1008626859765426e-07 * Pl.^3 + ...
        2.5564298545331497e-08 * Pl.^3 .* Vl + ...
        -2.150125796817501e-09 * Pl.^3 .* Vl.^2 + ...
        5.66075567906357e-11 * Pl.^3 .* Vl.^3;

    % Upper region: P >= 28
    idx_upper = P >= 28;
    Pu = P(idx_upper);
    Vu = V(idx_upper);

    force(idx_upper) = ...
        -1.301472592175583 + ...
        0.29936256569217357 * Vu + ...
        -0.023862645553737804 * Vu.^2 + ...
        0.000604424709842914 * Vu.^3 + ...
        0.028430427283555017 * Pu + ...
        -0.006798999839840506 * Pu .* Vu + ...
        0.0005532577424440186 * Pu .* Vu.^2 + ...
        -1.386998112663601e-05 * Pu .* Vu.^3 + ...
        -8.657617313618081e-05 * Pu.^2 + ...
        2.1124768936001382e-05 * Pu.^2 .* Vu + ...
        -1.3098984819064634e-06 * Pu.^2 .* Vu.^2 + ...
        2.7592710221785262e-08 * Pu.^2 .* Vu.^3 + ...
        4.2685499993615604e-08 * Pu.^3 + ...
        -9.090161175851121e-09 * Pu.^3 .* Vu + ...
        2.5267430874260785e-10 * Pu.^3 .* Vu.^2 + ...
        5.873211599828194e-12 * Pu.^3 .* Vu.^3;
   end
end


ForceFromPWMVoltagePlot

F = ForceFromPWMVoltage(1640, 12);
fprintf('Force at PWM = 1640 and Voltage = 12 is %.2f Kg f\n', F);

F = ForceFromPWMVoltage(1364, 12);
fprintf('Force at PWM = 1364 and Voltage = 12 is %.2f Kg f\n', F);

F = ForceFromPWMVoltage(1464, 14);
fprintf('Force at PWM = 1464 and Voltage = 14 is %.2f Kg f\n', F);

F = ForceFromPWMVoltage(1820, 18);
fprintf('Force at PWM = 1820 and Voltage = 18 is %.2f Kg f\n', F);

F = ForceFromPWMVoltage(1364, 16);
fprintf('Force at PWM = 1364 and Voltage = 16 is %.2f Kg f\n', F);

F = ForceFromPWMVoltage(1472, 16);
fprintf('Force at PWM = 1472 and Voltage = 16 is %.2f Kg f\n', F);

F = ForceFromPWMVoltage(1528, 16);
fprintf('Force at PWM = 1528 and Voltage = 16 is %.2f Kg f\n', F);
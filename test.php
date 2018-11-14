<?php

require 'autoload.php';

use LSTM\LstmSimpleUnit as lstm;
use LSTM\InputGate;
use LSTM\OutputGate;
use LSTM\ForgetGate;
use LSTM\LstmUnit;

function compare($gate, $inputs, $factors) {
    echo 'Inputs:'.PHP_EOL;
    foreach ($inputs as $key => $value) {
        echo 'Calculate:';
        $gate->setInputs($inputs);
        $gate->setFactors($factors);
        $y = $gate->calculate();
        $calculate = ($gate->derivativeInputs())[$key];
        echo $calculate;
        echo ',';
        $dinputs = $inputs;
        $dinputs[$key] = $dinputs[$key] + 0.001;
        $gate->setInputs($dinputs);
        $gate->setFactors($factors);
        $dy = $gate->calculate();
        echo 'Real:';
        $real = ($dy - $y) / 0.001;
        echo $real;
        echo ',Diff:';
        echo $calculate - $real;
        echo PHP_EOL;
    }
    echo 'Factors:'.PHP_EOL;
    foreach ($factors as $key => $value) {
        echo 'Calculate:';
        $gate->setInputs($inputs);
        $gate->setFactors($factors);
        $y = $gate->calculate();
        $y = $gate->getCellStatus();
        $calculate = ($gate->derivativeFactors())[$key];
        $calculate = ($gate->derivativeFactorsStatus())[$key];
        echo $calculate;
        echo ',';
        $dfactors = $factors;
        $dfactors[$key] = $factors[$key] + 0.00001;
        $gate->setInputs($inputs);
        $gate->setFactors($dfactors);
        $dy = $gate->calculate();
        $dy = $gate->getCellStatus();
        echo 'Real:';
        $real = ($dy - $y) / 0.00001;
        echo $real;
        echo ',Diff:';
        echo $calculate - $real;
        echo PHP_EOL;
    }
    if (!($gate instanceof lstm)) {
        return 0;
    }

    echo 'Status/Inputs:'.PHP_EOL;
    foreach ($inputs as $key => $value) {
        echo 'Calculate:';
        $gate->setInputs($inputs);
        $gate->setFactors($factors);
        $y = $gate->calculate();
        $y = $gate->getCellStatus();
        $calculate = ($gate->derivativeInputs())[$key];
        $calculate = ($gate->derivativeInputsStatus())[$key];
        echo $calculate;
        echo ',';
        $dinputs = $inputs;
        $dinputs[$key] = $dinputs[$key] + 0.001;
        $gate->setInputs($dinputs);
        $gate->setFactors($factors);
        $dy = $gate->calculate();
        $dy = $gate->getCellStatus();
        echo 'Real:';
        $real = ($dy - $y) / 0.001;
        echo $real;
        echo ',Diff:';
        echo $calculate - $real;
        echo PHP_EOL;
    }
    echo 'Status/Factors:'.PHP_EOL;
    foreach ($factors as $key => $value) {
        echo 'Calculate:';
        $gate->setInputs($inputs);
        $gate->setFactors($factors);
        $y = $gate->calculate();
        $y = $gate->getCellStatus();
        $calculate = ($gate->derivativeFactors())[$key];
        $calculate = ($gate->derivativeFactorsStatus())[$key];
        echo $calculate;
        echo ',';
        $dfactors = $factors;
        $dfactors[$key] = $factors[$key] + 0.00001;
        $gate->setInputs($inputs);
        $gate->setFactors($dfactors);
        $dy = $gate->calculate();
        $dy = $gate->getCellStatus();
        echo 'Real:';
        $real = ($dy - $y) / 0.00001;
        echo $real;
        echo ',Diff:';
        echo $calculate - $real;
        echo PHP_EOL;
    }
}

$inputs = [];
for ($i = 0; $i < 3; $i++) {
    $inputs[] = mt_rand() / mt_getrandmax();
}
$factors = [];
for ($i = 0; $i < 12; $i++) {
    $factors[] = mt_rand() / mt_getrandmax();
}

$gete = new lstm();

compare($gete, $inputs, $factors);

echo '------------------------------'.PHP_EOL;

$inputs = [];
for ($i = 0; $i < 12; $i++) {
    $inputs[] = $i % 3 > 0 ? 1 : 0;
}
$outputs = [];
for ($i = 0; $i < 12; $i++) {
    $outputs[] = $i % 3 > 0 ? 1 : 0;
}
echo 'Inputs:'.implode(' ', $inputs).PHP_EOL;
echo 'Outputs:'.implode(' ', $outputs).PHP_EOL;


$t = 15000;
$un = new LstmUnit();
$un->initFactors([], 'random');
$un->setSequence($inputs);
$un->setTargetSequence($outputs);
$un->run();
print_r($un->getLastOutputSequence());
$un->finiteSequenceOptimize(0.01, $t);
$h = $un->getHistory();
var_dump($h[$t-1]);
print_r($un->getLastOutputSequence());
file_put_contents('a.csv', implode(PHP_EOL, $h));
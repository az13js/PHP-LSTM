<?php
namespace LSTM;

/**
 * 遗忘门
 *
 * 算法：y=tanh(C)*sigmoid(w*h+u*x+b)。wub是参数，Chx是变量。
 *
 * @author az13js
 */
class OutputGate implements MathFunction
{
    /** @var float */
    private $w;
    /** @var float */
    private $u;
    /** @var float */
    private $b;
    /** @var float */
    private $C;
    /** @var float */
    private $h;
    /** @var float */
    private $x;

    /**
     * 设置输入
     *
     * 算法的输入有3个，C,h和x。h代表上一个LSTM单元的输出。
     *
     * @param array $inputs $inputs[0]:C,$inputs[1]:h,$inputs[2]:x
     * @return bool
     */
    public function setInputs(array $inputs): bool
    {
        if (!$this->verify($inputs, 3)) {
            throw new \Exception('In OutputGate::setInputs() '.serialize($inputs));
        }
        $this->C = $inputs[0];
        $this->h = $inputs[1];
        $this->x = $inputs[2];
        return true;
    }

    /**
     * 设置参数
     *
     * @param array $factors
     * @return bool
     */
    public function setFactors(array $factors): bool
    {
        if (!$this->verify($factors, 3)) {
            throw new \Exception('In OutputGate::setFactors() '.serialize($factors));
        }
        $this->w = $factors[0];
        $this->u = $factors[1];
        $this->b = $factors[2];
        return true;
    }

    /**
     * 获取输入的值
     *
     * @return array
     */
    public function getInputs(): array
    {
        return [$this->C, $this->h, $this->x];
    }

    /**
     * 获取设置的参数
     *
     * @return array
     */
    public function getFactors(): array
    {
        return [$this->w, $this->u, $this->b];
    }

    /**
     * 函数进行运算，得出最终的值
     *
     * @return float
     */
    public function calculate(): float
    {
        $this->verifyProperty();
        $w = $this->w;
        $u = $this->u;
        $b = $this->b;
        $C = $this->C;
        $h = $this->h;
        $x = $this->x;
        $sum = $w * $h + $u * $x + $b;
        $y = $this->tanh($C) * $this->sigmoid($sum);
        return $y;
    }

    /**
     * 已知的参数和输入下，返回输出对输入的偏导数
     *
     * @return array
     */
    public function derivativeInputs(): array
    {
        $this->verifyProperty();
        $w = $this->w;
        $u = $this->u;
        $b = $this->b;
        $C = $this->C;
        $h = $this->h;
        $x = $this->x;
        $sum = $w * $h + $u * $x + $b;
        $tanhC = $this->tanh($C);
        $dSigmoidsum = $this->derivativeSigmoid($sum);
        $dC = $this->derivativeTanh($C) * $this->sigmoid($sum);
        $dh = $tanhC * $dSigmoidsum * $w;
        $dx = $tanhC * $dSigmoidsum * $u;
        return [$dC, $dh, $dx];
    }

    /**
     * 在已知的参数和输入下，返回输出对参数的偏导数
     *
     * @return array
     */
    public function derivativeFactors(): array
    {
        $this->verifyProperty();
        $w = $this->w;
        $u = $this->u;
        $b = $this->b;
        $C = $this->C;
        $h = $this->h;
        $x = $this->x;
        $sum = $w * $h + $u * $x + $b;
        $tanhC = $this->tanh($C);
        $dSigmoidsum = $this->derivativeSigmoid($sum);
        $dw = $tanhC * $dSigmoidsum * $h;
        $du = $tanhC * $dSigmoidsum * $x;
        $db = $tanhC * $dSigmoidsum;
        return [$dw, $du, $db];
    }

    /**
     * 验证参数的正确性
     *
     * @param array $param 输入参数数组
     * @param int $length 输入参数应有的长度
     * @return bool 不符合长度以及元素不是数字则返回false
     */
    private function verify(array $param, int $length): bool
    {
        if ($length != count($param)) {
            return false;
        } else {
            foreach ($param as $v) {
                if (!is_numeric($v)) {
                    return false;
                }
            }
            return true;
        }
    }

    /**
     * 验证类属性的合法性
     *
     * @return void
     */
    private function verifyProperty(): void
    {
        if (
            !is_numeric($this->w) ||
            !is_numeric($this->u) ||
            !is_numeric($this->b) ||
            !is_numeric($this->C) ||
            !is_numeric($this->h) ||
            !is_numeric($this->x)
        ) {
            throw new \Exception('Error in OutputGate, property type error!');
        }
    }

    /**
     * Sigmoid函数
     *
     * @param float $x
     * @return float
     */
    private function sigmoid(float $x): float
    {
        return 1 / (1 + exp(-$x));
    }

    /**
     * Sigmoid的导数
     *
     * @param float $x
     * @return float
     */
    private function derivativeSigmoid(float $x): float
    {
        $y = 1 / (1 + exp(-$x));
        return $y * (1 - $y);
    }

    /**
     * tanh函数
     *
     * @param float $x
     * @return float
     */
    private function tanh(float $x): float
    {
        return tanh($x);
    }

    /**
     * tanh的导数
     *
     * @param float $x
     * @return float
     */
    private function derivativeTanh(float $x): float
    {
        $y = $this->tanh($x);
        return 1 - $y * $y;
    }
}

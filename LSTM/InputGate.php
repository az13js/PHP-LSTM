<?php
namespace LSTM;

/**
 * 输入门
 *
 * 算法（wub是参数，hx是变量）：
 * i=sigmoid(wi*h+ui*x+bi)
 * a=tanh(wa*h+ua*x+ba)
 * y=i*a
 *
 * @author az13js
 */
class InputGate implements MathFunction
{
    /** @var float */
    private $wi;
    /** @var float */
    private $ui;
    /** @var float */
    private $bi;
    /** @var float */
    private $wa;
    /** @var float */
    private $ua;
    /** @var float */
    private $ba;
    /** @var float */
    private $h;
    /** @var float */
    private $x;

    /**
     * 设置输入
     *
     * 算法的输入有两个，h和x。h代表上一个LSTM单元的输出。
     *
     * @param array $inputs $inputs[0]:h,$inputs[1]:x
     * @return bool
     */
    public function setInputs(array $inputs): bool
    {
        if (!$this->verify($inputs, 2)) {
            throw new \Exception('In InputGate::setInputs() '.serialize($inputs));
        }
        $this->h = $inputs[0];
        $this->x = $inputs[1];
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
        if (!$this->verify($factors, 6)) {
            throw new \Exception('In InputGate::setFactors() '.serialize($factors));
        }
        $this->wi = $factors[0];
        $this->ui = $factors[1];
        $this->bi = $factors[2];
        $this->wa = $factors[3];
        $this->ua = $factors[4];
        $this->ba = $factors[5];
        return true;
    }

    /**
     * 获取输入的值
     *
     * @return array
     */
    public function getInputs(): array
    {
        return [$this->h, $this->x];
    }

    /**
     * 获取设置的参数
     *
     * @return array
     */
    public function getFactors(): array
    {
        return [$this->wi, $this->ui, $this->bi, $this->wa, $this->ua, $this->ba];
    }

    /**
     * 函数进行运算，得出最终的值
     *
     * @return float
     */
    public function calculate(): float
    {
        $this->verifyProperty();
        $wi = $this->wi;
        $ui = $this->ui;
        $bi = $this->bi;
        $wa = $this->wa;
        $ua = $this->ua;
        $ba = $this->ba;
        $h = $this->h;
        $x = $this->x;
        $sumi = $wi * $h + $ui * $x + $bi;
        $suma = $wa * $h + $ua * $x + $ba;
        $y = $this->sigmoid($sumi) * $this->tanh($suma);
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
        $wi = $this->wi;
        $ui = $this->ui;
        $bi = $this->bi;
        $wa = $this->wa;
        $ua = $this->ua;
        $ba = $this->ba;
        $h = $this->h;
        $x = $this->x;
        $sumi = $wi * $h + $ui * $x + $bi;
        $suma = $wa * $h + $ua * $x + $ba;
        $i = $this->sigmoid($sumi);
        $a = $this->tanh($suma);
        $dsi = $this->derivativeSigmoid($sumi);
        $dta = $this->derivativeTanh($suma);
        // (i*a)'=i'*a+i*a' -> d(i*a) = di*a+i*da
        /**
        i=sigmoid(wi*h+ui*x+bi)
        a=tanh(wa*h+ua*x+ba)
        y=i*a
        **/
        // 求h的偏导数
        $di = $dsi * $wi;
        $da = $dta * $wa;
        $dh = $di * $a + $i * $da;
        // 求x的偏导数
        $di = $dsi * $ui;
        $da = $dta * $ua;
        $dx = $di * $a + $i * $da;
        return [$dh, $dx];
    }

    /**
     * 在已知的参数和输入下，返回输出对参数的偏导数
     *
     * @return array
     */
    public function derivativeFactors(): array
    {
        $this->verifyProperty();
        $wi = $this->wi;
        $ui = $this->ui;
        $bi = $this->bi;
        $wa = $this->wa;
        $ua = $this->ua;
        $ba = $this->ba;
        $h = $this->h;
        $x = $this->x;
        $sumi = $wi * $h + $ui * $x + $bi;
        $suma = $wa * $h + $ua * $x + $ba;
        $i = $this->sigmoid($sumi);
        $a = $this->tanh($suma);
        $dsi = $this->derivativeSigmoid($sumi);
        $dta = $this->derivativeTanh($suma);
        // (i*a)'=i'*a+i*a' -> d(i*a) = di*a+i*da
        /**
        i=sigmoid(wi*h+ui*x+bi)
        a=tanh(wa*h+ua*x+ba)
        y=i*a
        **/
        $dwi = $a * $dsi * $h;
        $dui = $a * $dsi * $x;
        $dbi = $a * $dsi;
        $dwa = $i * $dta * $h;
        $dua = $i * $dta * $x;
        $dba = $i * $dta;
        return [$dwi, $dui, $dbi, $dwa, $dua, $dba];
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
            !is_numeric($this->wi) ||
            !is_numeric($this->ui) ||
            !is_numeric($this->bi) ||
            !is_numeric($this->wa) ||
            !is_numeric($this->ua) ||
            !is_numeric($this->ba) ||
            !is_numeric($this->h) ||
            !is_numeric($this->x)
        ) {
            throw new \Exception('Error in InputGate, property type error!');
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

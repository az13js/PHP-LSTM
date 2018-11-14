<?php
namespace LSTM;

/**
 * LSTM总算法，由三个门合成
 *
 * @author az13js
 */
class LstmSimpleUnit implements MathFunction
{
    /** @var float 算法参数1 */
    private $wf;
    /** @var float 算法参数2 */
    private $uf;
    /** @var float 算法参数3 */
    private $bf;
    /** @var float 算法参数4 */
    private $wi;
    /** @var float 算法参数5 */
    private $ui;
    /** @var float 算法参数6 */
    private $bi;
    /** @var float 算法参数7 */
    private $wa;
    /** @var float 算法参数8 */
    private $ua;
    /** @var float 算法参数9 */
    private $ba;
    /** @var float 算法参数10 */
    private $wo;
    /** @var float 算法参数11 */
    private $uo;
    /** @var float 算法参数12 */
    private $bo;
    /** @var float 给定输入1 */
    private $C0;
    /** @var float 给定输入2 */
    private $h0;
    /** @var float 给定输入3 */
    private $x;
    /** @var float 输出1 */
    private $C;
    /** @var float 输出2 */
    private $h;

    /**
     * 设置输入变量
     *
     * @param array $inputs 长度为3的数组，分别是C_t-1,h_t-1,x_t
     * @return bool
     */
    public function setInputs(array $inputs): bool
    {
        if (!$this->verify($inputs, 3)) {
            throw new \Exception('In class LstmSimpleUnit, "inputs" param error. '.serialize($inputs));
        }
        $this->C0 = $inputs[0];
        $this->h0 = $inputs[1];
        $this->x = $inputs[2];
        return true;
    }

    /**
     * 设置参数
     *
     * @param array $factors 总共有12个参数
     * @return bool
     */
    public function setFactors(array $factors): bool
    {
        if (!$this->verify($factors, 12)) {
            throw new \Exception('In class LstmSimpleUnit, "factors" param error. '.serialize($factors));
        }
        $this->wf = $factors[0];
        $this->uf = $factors[1];
        $this->bf = $factors[2];
        $this->wi = $factors[3];
        $this->ui = $factors[4];
        $this->bi = $factors[5];
        $this->wa = $factors[6];
        $this->ua = $factors[7];
        $this->ba = $factors[8];
        $this->wo = $factors[9];
        $this->uo = $factors[10];
        $this->bo = $factors[11];
        return true;
    }

    /**
     * 获取输入的值
     *
     * @return array
     */
    public function getInputs(): array
    {
        $inputs = [];
        $inputs[] = $this->C0;
        $inputs[] = $this->h0;
        $inputs[] = $this->x;
        return $inputs;
    }

    /**
     * 获取设置的参数
     *
     * @return array
     */
    public function getFactors(): array
    {
        $factors = [];
        $factors[] = $this->wf;
        $factors[] = $this->uf;
        $factors[] = $this->bf;
        $factors[] = $this->wi;
        $factors[] = $this->ui;
        $factors[] = $this->bi;
        $factors[] = $this->wa;
        $factors[] = $this->ua;
        $factors[] = $this->ba;
        $factors[] = $this->wo;
        $factors[] = $this->uo;
        $factors[] = $this->bo;
        return $factors;
    }

    /**
     * 运行LSTM，计算输出的值，并更新内部状态C
     *
     * @return float
     */
    public function calculate(): float
    {
        $this->verifyProperty();
        $C0 = $this->C0;
        $h0 = $this->h0;
        $x = $this->x;
        $wf = $this->wf;
        $uf = $this->uf;
        $bf = $this->bf;
        $wi = $this->wi;
        $ui = $this->ui;
        $bi = $this->bi;
        $wa = $this->wa;
        $ua = $this->ua;
        $ba = $this->ba;
        $wo = $this->wo;
        $uo = $this->uo;
        $bo = $this->bo;
        $forgetGate = new ForgetGate();
        $forgetGate->setInputs([$h0, $x]);
        $forgetGate->setFactors([$wf, $uf, $bf]);
        $f = $forgetGate->calculate();
        $inputGate = new InputGate();
        $inputGate->setInputs([$h0, $x]);
        $inputGate->setFactors([$wi, $ui, $bi, $wa, $ua, $ba]);
        $i = $inputGate->calculate();
        $C = $C0 * $f + $i;
        $outputGate = new OutputGate();
        $outputGate->setInputs([$C, $h0, $x]);
        $outputGate->setFactors([$wo, $uo, $bo]);
        $h = $outputGate->calculate();
        $this->C = $C;
        $this->h = $h;
        return $h;
    }

    /**
     * 返回细胞内部状态C
     *
     * @return float
     */
    public function getCellStatus(): float
    {
        return $this->C;
    }

    /**
     * 已知的参数和输入下，返回输出h对输入C0,h0,x的偏导数
     *
     * @return array
     */
    public function derivativeInputs(): array
    {
        $this->verifyProperty();
        $C0 = $this->C0;
        $h0 = $this->h0;
        $x = $this->x;
        $wf = $this->wf;
        $uf = $this->uf;
        $bf = $this->bf;
        $wi = $this->wi;
        $ui = $this->ui;
        $bi = $this->bi;
        $wa = $this->wa;
        $ua = $this->ua;
        $ba = $this->ba;
        $wo = $this->wo;
        $uo = $this->uo;
        $bo = $this->bo;
        $forgetGate = new ForgetGate();
        $forgetGate->setInputs([$h0, $x]);
        $forgetGate->setFactors([$wf, $uf, $bf]);
        $inputGate = new InputGate();
        $inputGate->setInputs([$h0, $x]);
        $inputGate->setFactors([$wi, $ui, $bi, $wa, $ua, $ba]);
        $f = $forgetGate->calculate();
        $i = $inputGate->calculate();
        $C = $C0 * $f + $i;
        $outputGate = new OutputGate();
        $outputGate->setInputs([$C, $h0, $x]);
        $outputGate->setFactors([$wo, $uo, $bo]);

        $dInputsOutputGate = $outputGate->derivativeInputs();
        $dInputsForgetGate = $forgetGate->derivativeInputs();
        $dInputsInputGate = $inputGate->derivativeInputs();

        $dC0 = $dInputsOutputGate[0] * $f;
        $dh0 = $dInputsOutputGate[1] + $dInputsOutputGate[0] * ($C0 * $dInputsForgetGate[0] + $dInputsInputGate[0]);
        $dx = $dInputsOutputGate[2] + $dInputsOutputGate[0] * ($C0 * $dInputsForgetGate[1] + $dInputsInputGate[1]);
        return [$dC0, $dh0, $dx];
    }

    /**
     * 在已知的参数和输入下，返回输出h对12个参数的偏导数
     *
     * @return array
     */
    public function derivativeFactors(): array
    {
        $this->verifyProperty();
        $C0 = $this->C0;
        $h0 = $this->h0;
        $x = $this->x;
        $wf = $this->wf;
        $uf = $this->uf;
        $bf = $this->bf;
        $wi = $this->wi;
        $ui = $this->ui;
        $bi = $this->bi;
        $wa = $this->wa;
        $ua = $this->ua;
        $ba = $this->ba;
        $wo = $this->wo;
        $uo = $this->uo;
        $bo = $this->bo;
        $forgetGate = new ForgetGate();
        $forgetGate->setInputs([$h0, $x]);
        $forgetGate->setFactors([$wf, $uf, $bf]);

        $inputGate = new InputGate();
        $inputGate->setInputs([$h0, $x]);
        $inputGate->setFactors([$wi, $ui, $bi, $wa, $ua, $ba]);

        $f = $forgetGate->calculate();
        $i = $inputGate->calculate();
        $C = $C0 * $f + $i;

        $outputGate = new OutputGate();
        $outputGate->setInputs([$C, $h0, $x]);
        $outputGate->setFactors([$wo, $uo, $bo]);

        $dInputsOutputGate = $outputGate->derivativeInputs();

        $dFactorsForgetGete = $forgetGate->derivativeFactors();
        $dFactorsInputGete = $inputGate->derivativeFactors();
        $dFactorsOutputGete = $outputGate->derivativeFactors();

        $dwf = $dInputsOutputGate[0] * $C0 * $dFactorsForgetGete[0];
        $duf = $dInputsOutputGate[0] * $C0 * $dFactorsForgetGete[1];
        $dbf = $dInputsOutputGate[0] * $C0 * $dFactorsForgetGete[2];
        $dwi = $dInputsOutputGate[0] * $dFactorsInputGete[0];
        $dui = $dInputsOutputGate[0] * $dFactorsInputGete[1];
        $dbi = $dInputsOutputGate[0] * $dFactorsInputGete[2];
        $dwa = $dInputsOutputGate[0] * $dFactorsInputGete[3];
        $dua = $dInputsOutputGate[0] * $dFactorsInputGete[4];
        $dba = $dInputsOutputGate[0] * $dFactorsInputGete[5];
        $dwo = $dFactorsOutputGete[0];
        $duo = $dFactorsOutputGete[1];
        $dbo = $dFactorsOutputGete[2];
        return [$dwf, $duf, $dbf, $dwi, $dui, $dbi, $dwa, $dua, $dba, $dwo, $duo, $dbo];
    }

    /**
     * 已知的参数和输入下，返回输出C对输入C0,h0,x的偏导数
     *
     * @return array
     */
    public function derivativeInputsStatus(): array
    {
        $this->verifyProperty();
        $C0 = $this->C0;
        $h0 = $this->h0;
        $x = $this->x;
        $wf = $this->wf;
        $uf = $this->uf;
        $bf = $this->bf;
        $wi = $this->wi;
        $ui = $this->ui;
        $bi = $this->bi;
        $wa = $this->wa;
        $ua = $this->ua;
        $ba = $this->ba;
        $wo = $this->wo;
        $uo = $this->uo;
        $bo = $this->bo;
        $forgetGate = new ForgetGate();
        $forgetGate->setInputs([$h0, $x]);
        $forgetGate->setFactors([$wf, $uf, $bf]);
        $inputGate = new InputGate();
        $inputGate->setInputs([$h0, $x]);
        $inputGate->setFactors([$wi, $ui, $bi, $wa, $ua, $ba]);

        $f = $forgetGate->calculate();
        $i = $inputGate->calculate();
        $C = $C0 * $f + $i;

        $outputGate = new OutputGate();
        $outputGate->setInputs([$C, $h0, $x]);
        $outputGate->setFactors([$wo, $uo, $bo]);

        $dC0 = $f;
        $dInputsForgetGate = $forgetGate->derivativeInputs();
        $dInputsInputGate = $inputGate->derivativeInputs();
        $dh0 = $C0 * $dInputsForgetGate[0] + $dInputsInputGate[0];
        $dx = $C0 * $dInputsForgetGate[1] + $dInputsInputGate[1];
        return [$dC0, $dh0, $dx];
    }

    /**
     * 返回输出C对各个参数的导数
     *
     * @return array
     */
    public function derivativeFactorsStatus(): array
    {
        $this->verifyProperty();
        $C0 = $this->C0;
        $h0 = $this->h0;
        $x = $this->x;
        $wf = $this->wf;
        $uf = $this->uf;
        $bf = $this->bf;
        $wi = $this->wi;
        $ui = $this->ui;
        $bi = $this->bi;
        $wa = $this->wa;
        $ua = $this->ua;
        $ba = $this->ba;
        $wo = $this->wo;
        $uo = $this->uo;
        $bo = $this->bo;
        $forgetGate = new ForgetGate();
        $forgetGate->setInputs([$h0, $x]);
        $forgetGate->setFactors([$wf, $uf, $bf]);
        $inputGate = new InputGate();
        $inputGate->setInputs([$h0, $x]);
        $inputGate->setFactors([$wi, $ui, $bi, $wa, $ua, $ba]);

        $dFactorsForgetGete = $forgetGate->derivativeFactors();
        $dFactorsInputGete = $inputGate->derivativeFactors();

        $dwf = $C0 * $dFactorsForgetGete[0];
        $duf = $C0 * $dFactorsForgetGete[1];
        $dbf = $C0 * $dFactorsForgetGete[2];
        $dwi = $dFactorsInputGete[0];
        $dui = $dFactorsInputGete[1];
        $dbi = $dFactorsInputGete[2];
        $dwa = $dFactorsInputGete[3];
        $dua = $dFactorsInputGete[4];
        $dba = $dFactorsInputGete[5];
        $dwo = 0;
        $duo = 0;
        $dbo = 0;
        return [$dwf, $duf, $dbf, $dwi, $dui, $dbi, $dwa, $dua, $dba, $dwo, $duo, $dbo];
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
            !is_numeric($this->wf) ||
            !is_numeric($this->uf) ||
            !is_numeric($this->bf) ||
            !is_numeric($this->wo) ||
            !is_numeric($this->uo) ||
            !is_numeric($this->bo) ||
            !is_numeric($this->wi) ||
            !is_numeric($this->ui) ||
            !is_numeric($this->bi) ||
            !is_numeric($this->wa) ||
            !is_numeric($this->ua) ||
            !is_numeric($this->ba) ||
            !is_numeric($this->C0) ||
            !is_numeric($this->h0) ||
            !is_numeric($this->x)
        ) {
            throw new \Exception('Error in LstmSimpleUnit, property type error!');
        }
    }
}

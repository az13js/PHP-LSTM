<?php
namespace LSTM;

/**
 * LSTM单元
 *
 * LSTM单元对一个序列进行运算后得到一个长度相等的新序列。LSTM单元可以通过反向传播梯度下降的
 * 方法修正误差，从而得到序列到序列的拟合能力。
 *
 * @author az13js
 */
class LstmUnit
{
    /** @var array 输入序列 */
    private $sequence;
    /** @var array 按照顺序存放的LSTM参数 */
    private $factors;
    /** @var array 如果有，就是LSTM算出来的输出序列 */
    private $hSequence;
    /** @var array 优化的目标序列 */
    private $targetSequence;
    /** @var array 考虑之前的输入情况，每一个参数对当前输出的影响 */
    private $effectH;
    /** @var array 考虑之前的输入情况，每一个参数对当前状态的影响 */
    private $effectC;
    /** @var array */
    private $history;

    /**
     * 设置输入序列的值
     *
     * @param array $sequence
     * @return bool
     */
    public function setSequence(array $sequence): bool
    {
        if (!is_array($sequence)) {
            throw new \Exception('setSequence() recive a array! '.serialize($sequence));
        }
        $this->sequence = [];
        foreach ($sequence as $value) {
            if (!is_numeric($value)) {
                throw new \Exception('element in $sequence is not a number format! '.serialize($sequence));
            }
            $this->sequence[] = $value;
        }
        return true;
    }

    /**
     * 返回设置的序列
     *
     * @return array
     */
    public function getSequence(): array
    {
        return $this->sequence;
    }

    /**
     * 初始化参数
     *
     * 可以这样来随机初始化：initFactors([], 'random');
     *
     * @param array $factors 长度为12的数组，下标0~11的参数含义是wf,uf,bf,wi,ui,bi,wa,ua,ba,wo,uo,bo
     * @param string $type 默认值是空字符串，当此值等于random时会忽略第一个参数，对参数进行随机赋值
     * @return bool
     */
    public function initFactors(array $factors = [], string $type = ''): bool
    {
        if ('random' != $type) {
            if (!is_array($factors)) {
                throw new \Exception('init() recive a array! '.serialize($factors));
            }
            if (12 != count($factors)) {
                throw new \Exception('12 != count($factors)! '.serialize($factors));
            }
            $this->factors = [];
            foreach ($factors as $value) {
                if (!is_numeric($value)) {
                    throw new \Exception('element in $factors is not a number format! '.serialize($factors));
                }
                $this->factors[] = $value;
            }
        } else {
            $this->factors = [];
            for ($i = 0; $i < 12; $i++) {
                $sign = mt_rand() / mt_getrandmax() < 0.5 ? -1 : 1;
                $this->factors[] = $sign * mt_rand() / mt_getrandmax();
            }
        }
        return true;
    }

    /**
     * 返回LSTM参数
     *
     * @return array 下标0~11的参数含义是wf,uf,bf,wi,ui,bi,wa,ua,ba,wo,uo,bo
     */
    public function getFactors(): array
    {
        return $this->factors;
    }

    /**
     * 根据已知的参数和输入序列，计算输出序列
     *
     * @return bool
     */
    public function run(): bool
    {
        if (!is_array($this->sequence)) {
            throw new \Exception('$this->sequence is not a array! '.serialize($this->sequence));
        }
        $lstm = new LstmSimpleUnit();
        $lstm->setFactors($this->factors);
        $this->hSequence = [];
        $h = 0;
        $C = 0;
        foreach ($this->sequence as $x) {
            $lstm->setInputs([$C, $h, $x]);
            $h = $lstm->calculate();
            $C = $lstm->getCellStatus();
            $this->hSequence[] = $h;
        }
        return true;
    }

    /**
     * 返回最后一次进行run运算后的输出序列
     *
     * @return array
     */
    public function getLastOutputSequence(): array
    {
        return $this->hSequence;
    }

    /**
     * 设置目标序列。
     *
     * 目标序列是用于逼近的目标，梯度下降会使用到。
     *
     * @param array $sequence 前面不需要的部分可以设置为任意数，不能为非数字。
     * @return bool
     */
    public function setTargetSequence(array $sequence): bool
    {
        if (!is_array($sequence)) {
            throw new \Exception('$sequence is not a array! '.serialize($sequence));
        }
        $this->targetSequence = [];
        foreach ($sequence as $v) {
            if (!is_numeric($v)) {
                throw new \Exception('$sequence is not a number array! '.serialize($sequence));
            }
            $this->targetSequence[] = $v;
        }
        return true;
    }

    /**
     * 获取设置的目标序列。
     *
     * @return array
     */
    public function getTargetSequence(): array
    {
        return $this->targetSequence;
    }

    /**
     * 利用梯度下降和反向传播，优化LSTM的参数。
     *
     * 此方法针对有限长度的序列。即数组可以放入类属性里，统一优化的。
     *
     * @param float $alpha 学习率
     * @param int $count 设置允许循环的最大次数
     * @param float $error 误差小于这个值的时候停止循环
     * @param int $len 有效长度。输出序列倒数多少位是有用的。0代表全部有用
     * @return bool
     */
    public function finiteSequenceOptimize(float $alpha = 0.001, int $count = 10, float $error = 0, int $len = 0): bool
    {
        $sequenceSize = count($this->sequence, COUNT_NORMAL);
        if ($sequenceSize != count($this->targetSequence, COUNT_NORMAL)) {
            throw new \Exception('target size != sequence size sequence:'.$sequenceSize.',target:'.count($this->targetSequence, COUNT_NORMAL));
        }
        if ($len > $sequenceSize) {
            throw new \Exception('length > '.$sequenceSize);
        }
        $lstm = new LstmSimpleUnit();
        $this->history = [];
        for ($i = 0; $i < $count; $i++) {
            $dfactors = [];
            for ($j = 0; $j < 12; $j++) {
                $dfactors[] = 0;
            }
            /*
             * 根据输入和目标输出序列，计算12个参数的偏导数
             */
            $lstm->setFactors($this->factors);
            $h = 0;
            $C = 0;
            $this->hSequence = [];
            $this->effectH = [];
            $this->effectC = [];
            for ($j = 0; $j < $sequenceSize; $j++) {
                $lstm->setInputs([$C, $h, $this->sequence[$j]]);
                /*
                 * 计算h和C，这两个参数在下一个循环里会参与计算
                 */
                $h = $lstm->calculate();
                $C = $lstm->getCellStatus();
                $this->hSequence[] = $h;
                $this->effectH[] = $lstm->derivativeFactors();/*[$dwf, $duf, $dbf, $dwi, $dui, $dbi, $dwa, $dua, $dba, $dwo, $duo, $dbo];*/
                $this->effectC[] = $lstm->derivativeFactorsStatus();/*[$dwf, $duf, $dbf, $dwi, $dui, $dbi, $dwa, $dua, $dba, $dwo, $duo, $dbo];*/
                if ($j > 0) {
                    $dhdInputs = $lstm->derivativeInputs();/*[$dC0, $dh0, $dx];*/
                    $dCdInputs = $lstm->derivativeInputsStatus();/*[$dC0, $dh0, $dx];*/
                    /*
                     * h和C对p的偏导数数组具有相同的结构，放在一起算
                     */
                    foreach ($this->effectH[$j] as $k => $p) {
                        $this->effectH[$j][$k] += $dhdInputs[0] * $this->effectC[$j - 1][$j] + $dhdInputs[1] * $this->effectH[$j - 1][$k];
                        $this->effectC[$j][$k] += $dCdInputs[1] * $this->effectH[$j - 1][$j] + $dCdInputs[0] * $this->effectC[$j - 1][$k];
                    }
                }
                $ignore = 0;
                if ($len > 0) {
                    $ignore = $sequenceSize - $len;
                }
                if ($j < $ignore) {
                    continue;
                }
                $dMSEdh = 2 * ($h - $this->targetSequence[$j]) / $sequenceSize;
                foreach ($dfactors as $k => $v) {
                    $dfactors[$k] += $dMSEdh * $this->effectH[$j][$k];
                }
            }
            /*
             * 到此为止，参数对误差函数的影响计算完成。下面计算误差和更新参数。
             */
            $ignore = 0;
            if ($len > 0) {
                $ignore = $sequenceSize - $len;
            }
            $MSE = 0;
            foreach ($this->hSequence as $k => $v) {
                if ($k < $ignore) {
                    continue;
                }
                $MSE += pow($v - $this->targetSequence[$k], 2);
            }
            $MSE /= $sequenceSize;
            $this->history[] = $MSE;
            if ($MSE < $error) {
                break;
            }
            foreach ($this->factors as $k => $v) {
                $this->factors[$k] -= $dfactors[$k] * $alpha;
            }
        }
        return true;
    }

    /**
     * 获取历史误差
     *
     * @return array
     */
    public function getHistory(): array
    {
        return $this->history;
    }
}

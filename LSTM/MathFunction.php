<?php
namespace LSTM;

/**
 * 数学函数
 *
 * 设置函数的输入和参数，然后调用其它方法获取函数的运算结果。
 *
 * @author az13js
 */
interface MathFunction
{
    /**
     * 设置输入
     *
     * @param array $inputs
     * @return bool
     */
    public function setInputs(array $inputs): bool;

    /**
     * 设置参数
     *
     * @param array $factors
     * @return bool
     */
    public function setFactors(array $factors): bool;

    /**
     * 获取输入的值
     *
     * @return array
     */
    public function getInputs(): array;

    /**
     * 获取设置的参数
     *
     * @return array
     */
    public function getFactors(): array;

    /**
     * 函数进行运算，得出最终的值
     *
     * @return float
     */
    public function calculate(): float;

    /**
     * 已知的参数和输入下，返回输出对输入的偏导数
     *
     * @return array
     */
    public function derivativeInputs(): array;

    /**
     * 在已知的参数和输入下，返回输出对参数的偏导数
     *
     * @return array
     */
    public function derivativeFactors(): array;
}

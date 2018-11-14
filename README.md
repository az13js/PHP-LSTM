# PHP LSTM

> JUST FOR FUN.

LSTM是长短时记忆网络，它是一种用于时序序列处理的机器学习算法。LSTM在语音信号处理、文本序列预测、语句情感分析和翻译等自然语言处理的场景中有着广泛的应用。

LSTM具有以下基本特点：

1. 一个LSTM单元由12个参数决定；
2. 可以对一个任意长有序序列进行运算得到一个长度相等的序列；
3. 利用误差反向传播和梯度下降的方法可以在已有两个序列的情况下拟合一个LSTM单元；
4. 可以只使用LSTM输出的最后几位，也可以只利用这几位来进行拟合。

LSTM每一次计算时根据当前的输入x、上一次的输出h和上一次计算得到的状态参数C来算出当前的输出h和更新状态C。换言之LSTM算法需要x、h0、C0作为输入，输出内容有h、C，而h和C会作为下一次的输入参数与下一个x一起在算法里进行计算。

与RNN相比，LSTM连续计算的过程中除了单元的输出h会参与下一次的计算外还多了一个状态参数C。C并不直接输出给用户，只是充当内部参数进行传递，因此中间参数C在梯度下降时可以不直接被输入输出所影响而改变。综上所述，C的存在使得LSTM能比RNN更好地提供长时程的依赖能力。

# 主要用法示例

机器学习用法

    // 创建LSTM
	$un = new LstmUnit();
    // 随机初始化参数
	$un->initFactors([], 'random');
    // 设置输入数组
	$un->setSequence($inputs);
    // 设置期望输出
	$un->setTargetSequence($outputs);
    // 梯度下降
	$un->finiteSequenceOptimize(0.001, 20000);
    // 获取梯度下降时损失函数的变化
	$h = $un->getHistory();

用来预测

	$un->setSequence($inputs);
	$un->run();
	print_r($un->getLastOutputSequence());

详细说明参考文件 ```LSTM/LstmUnit.php``` 中的注释。

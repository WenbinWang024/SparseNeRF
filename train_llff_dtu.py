# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training script for RegNerf."""

import functools
import gc
import time

from absl import app
import flax
from flax.metrics import tensorboard
from flax.training import checkpoints
from internal import configs, datasets_depth_llff_dtu, math, models, utils, vis  # pylint: disable=g-multiple-import
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from skimage.metrics import structural_similarity
from jax import jit

configs.define_common_flags()
jax.config.parse_flags_with_absl()

TIME_PRECISION = 1000  # Internally represent integer times in milliseconds.


@flax.struct.dataclass
class TrainStats:
  """Collection of stats for logging."""
  loss: float
  losses: float
  losses_georeg: float
  disp_mses: float
  normal_maes: float
  weight_l2: float
  psnr: float
  psnrs: float
  grad_norm: float
  grad_abs_max: float
  grad_norm_clipped: float
  all_depth_loss: float

# 这个函数计算给定变量树（tree）中所有元素的总和。在 JAX 中，模型的参数通常存储在嵌套的数据结构中，tree_sum 通过 tree_reduce 函数对这些参数进行累加操作。
def tree_sum(tree):
  return jax.tree_util.tree_reduce(lambda x, y: x + y, tree, initializer=0) # x1+x2+x3+...


# 这个函数计算变量树中所有参数的 L2 范数。首先使用 tree_map 将每个元素平方，然后用 tree_sum 求和，最后取平方根。
def tree_norm(tree):
  return jnp.sqrt(tree_sum(jax.tree_map(lambda x: jnp.sum(x**2), tree))) # sqrt((x1^2+x2^2+...xn^2))

# 这是训练过程中的核心函数，执行一个优化步骤。它包括损失函数的定义、前向和后向传播、以及模型状态的更新。函数涉及到复杂的计算，包括渲染、损失计算、梯度计算和参数更新。
def train_step(
    model,
    config,
    rng,
    state,
    batch,
    learning_rate,
    resample_padding,
    tvnorm_loss_weight,
    step,
):
  """One optimization step.

  Args:
    model: The linen model.
    config: The configuration.
    rng: jnp.ndarray, random number generator.
    state: utils.TrainState, state of the model/optimizer.
    batch: dict, a mini-batch of data for training.
    learning_rate: float, real-time learning rate.
    resample_padding: float, the histogram padding to use when resampling.
    tvnorm_loss_weight: float, tvnorm loss weight.

  Returns:
    A tuple (new_state, stats, rng) with
      new_state: utils.TrainState, new training state.
      stats: list. [(loss, psnr), (loss_coarse, psnr_coarse)].
      rng: jnp.ndarray, updated random number generator.
  """
  rng, key, key2 = random.split(rng, 3)
#   print('step:', step)



  # 这个函数接收一个参数 variables，它包含了模型的所有可训练参数。
  def loss_fn(variables): # contains global vars of train_step, such as batch...

    # 计算权重 L2 正则化损失，这里首先对所有权重进行平方，然后求和，得到总的平方和。随后，计算所有权重的元素总数，最后将权重的平方和除以元素总数，得到权重的 L2 正则化损失。
    weight_l2 = (
        tree_sum(jax.tree_map(lambda z: jnp.sum(z**2), variables)) / tree_sum(
            jax.tree_map(lambda z: jnp.prod(jnp.array(z.shape)), variables)))   # (x1^2+x2^2+...xn^2)/n

    # 应用模型并获取渲染结果，这一行调用模型的 apply 方法，传入当前的参数、随机数生成器的键（如果配置了随机化）、当前批次的光线数据，以及其他控制渲染的参数。这将返回一个包含多个渲染结果的列表（每个渲染结果可能对应一个不同的网络级别或视角）
    renderings = model.apply(
        variables,
        key if config.randomized else None, # randomized is True
        batch['rays'],
        resample_padding=resample_padding,
        compute_extras=(config.compute_disp_metrics or
                        config.compute_normal_metrics))

    # 初始化损失和度量列表，这里根据批次数据中包含的损失权重来初始化 lossmult，如果配置禁用了多尺度损失，则将其设置为全1。
    lossmult = batch['rays'].lossmult
    if config.disable_multiscale_loss: #False
      lossmult = jnp.ones_like(lossmult)

    # 遍历每个渲染结果计算损失
    losses = []
    disp_mses = []
    normal_maes = []

    all_depth_loss = []

    # 对每个渲染结果进行处理，接下来的代码块遍历每个渲染结果，计算 RGB 损失、深度损失等，并可能计算视差 MSE 和法线 MAE（如果相应的指标计算被激活）
  
    for rendering in renderings:
      
      # 计算 RGB 损失
      numer = (lossmult * (rendering['rgb'] - batch['rgb'][Ellipsis, :3])**2).sum()
      denom = lossmult.sum()
      losses.append(numer / denom)

      # 计算深度相关的损失，这部分计算与深度预测有关的损失。例如，使用一定的间隙（margin）来确保预测的深度值之间保持一定的差异。
      depth = rendering['distance_mean']
      depth = depth.reshape(-1,4).transpose() # two pairs of points, so here is 4


      margin1 = 1e-4
      margin2 = 1e-4
      depth_loss0_0 = jnp.mean(jnp.maximum(depth[0,:]-depth[1,:]+margin1,0)) ###
      depth_loss0_1 = jnp.mean(jnp.maximum(jnp.abs(depth[0,:]-depth[2,:])-margin2,0)) 
      depth_loss0_2 = jnp.mean(jnp.maximum(jnp.abs(depth[1,:]-depth[3,:])-margin2,0)) 

      # 总结所有损失并返回，根据配置不同，深度损失的权重会有所不同。最终将这些损失汇总并添加到 all_depth_loss 列表中
      if config.dataset_loader=="llff":
        depth_loss = depth_loss0_0+(depth_loss0_1+depth_loss0_2)*0.01
      else:
        depth_loss = depth_loss0_0+(depth_loss0_1+depth_loss0_2)*1.0

      all_depth_loss.append(depth_loss)

      # 计算视差和法线度量的条件判断，这部分代码在配置文件指定计算视差和法线度量时才执行。视差（disp）通过渲染的平均距离来计算，然后与真实视差数据计算均方误差。法线误差（normal_mae）通过计算预测和实际法线之间的角度差异来计算平均绝对误差。
      if config.compute_disp_metrics: # False
        # Using mean to compute disparity, but other distance statistics can be
        # used instead.
        disp = 1 / (1 + rendering['distance_mean'])
        disp_mses.append(((disp - batch['disps'])**2).mean())
      if config.compute_normal_metrics: # False
        one_eps = 1 - jnp.finfo(jnp.float32).eps
        normal_mae = jnp.arccos(
            jnp.clip(
                jnp.sum(batch['normals'] * rendering['normals'], axis=-1),
                -one_eps, one_eps)).mean()
        normal_maes.append(normal_mae)


    # 计算基于 patch 的几何正则化损失，这一部分代码在配置中启用了深度 TV-Norm 损失或者几何正则化衰减时执行。它对随机选取的光线进行渲染，然后计算基于 patch 的总变差（TV）规范化损失。这有助于保持渲染的深度信息的局部一致性和平滑性。
    render_random_rays = ((config.depth_tvnorm_loss_mult != 0.0) or #depth_tvnorm_loss_mult=0.1
                          (config.depth_tvnorm_decay)) # depth_tvnorm_decay is True
    
    ########## patch based
    if render_random_rays: # True
      losses_georeg = []
      renderings_random = model.apply(
          variables,
          key2 if config.randomized else None,
          batch['rays_random'],
          resample_padding=resample_padding,
          compute_extras=True)
      ps = config.patch_size
      reshape_to_patch = lambda x, dim: x.reshape(-1, ps, ps, dim)
      for rendering in renderings_random:
        if config.depth_tvnorm_loss_mult != 0.0 or config.depth_tvnorm_decay:
          depth = reshape_to_patch(rendering[config.depth_tvnorm_selector], 1)
          weighting = jax.lax.stop_gradient(reshape_to_patch(rendering['acc'],1)[:, :-1, :-1]) * config.depth_tvnorm_mask_weight
          losses_georeg.append(math.compute_tv_norm(depth, config.depth_tvnorm_type, weighting).mean())
        else:
          losses_georeg.append(0.0)

    # 计算总损失并返回，最终计算的总损失是基于多种损失类型的加权和。根据不同的数据集配置（如 llff 或 dtu），损失的组合方式略有不同。最终，这个函数返回计算得到的总损失和包含所有损失组件的元组，用于后续的梯度计算和优化步骤。这种损失组合方式有助于调整模型对不同任务的响应，确保在颜色保真度、几何一致性和其他方面的性能平衡。
    losses = jnp.array(losses)
    losses_georeg = jnp.array(losses_georeg)
    disp_mses = jnp.array(disp_mses) # empty
    normal_maes = jnp.array(normal_maes) # empty
    all_depth_loss = jnp.array(all_depth_loss) 

    if config.dataset_loader=="llff":
      print('######################################### lwq llff')
      loss = losses[-1] + config.coarse_loss_mult * jnp.sum(losses[:-1]) + config.weight_decay_mult * weight_l2+0.5*all_depth_loss[-1]
    else:
      print('######################################### lwq dtu')
      loss = losses[-1] + config.coarse_loss_mult * jnp.sum(losses[:-1]) + config.weight_decay_mult * weight_l2+0.2*all_depth_loss[-1]+\
          ((tvnorm_loss_weight if config.depth_tvnorm_decay else
            config.depth_tvnorm_loss_mult) *  losses_georeg[-1] + \
           config.coarse_loss_mult * jnp.sum(losses_georeg[:-1]))
  

    return loss, (losses, disp_mses, normal_maes, weight_l2, losses_georeg, all_depth_loss)
  
  ##############
  #target – the object to be optimized. This is typically a variable dict returned by flax.linen.Module.init()

  # 这行代码使用 JAX 的 value_and_grad 函数来同时计算损失和梯度。loss_fn 是损失函数，state.optimizer.target 包含模型的参数。has_aux=True 表明损失函数还返回一些额外的数据（存储在 loss_aux 中）。
  (loss, loss_aux), grad = (jax.value_and_grad(loss_fn, has_aux=True)(          ##########?
      state.optimizer.target))

  # 平均梯度和损失，这行代码将 loss_aux 中的数据解构到相应的变量中，这些数据包括不同类型的损失和度量。
  (losses, disp_mses, normal_maes, weight_l2, losses_georeg, all_depth_loss) = loss_aux

  # 使用 jax.lax.pmean 函数对梯度和所有损失进行跨设备的平均。这是在分布式训练中同步不同设备上计算结果的常用做法。
  grad = jax.lax.pmean(grad, axis_name='batch') #Compute an all-reduce mean on x over the pmapped axis axis_name.
  losses = jax.lax.pmean(losses, axis_name='batch')
  disp_mses = jax.lax.pmean(disp_mses, axis_name='batch')
  normal_maes = jax.lax.pmean(normal_maes, axis_name='batch')
  weight_l2 = jax.lax.pmean(weight_l2, axis_name='batch')
  losses_georeg = jax.lax.pmean(losses_georeg, axis_name='batch')
  all_depth_loss = jax.lax.pmean(all_depth_loss, axis_name='batch')
  

  # 检查梯度中的 NaN 值并替换，如果启用了 NaN 检查，则使用 jax.tree_map 和 jnp.nan_to_num 将梯度中的 NaN 值替换为数值0。
  if config.check_grad_for_nans: # False
    grad = jax.tree_map(jnp.nan_to_num, grad)

  # 限制梯度值的范围，这行代码限制梯度值的最大绝对值，防止梯度爆炸。
  if config.grad_max_val > 0: #grad_max_val=0.1
    grad = jax.tree_map(
        lambda z: jnp.clip(z, -config.grad_max_val, config.grad_max_val), grad) # clip(-0.1, 0.1)

  # 计算梯度的最大绝对值和 L2 范数，计算梯度的最大绝对值和 L2 范数，用于监控训练过程中梯度的大小。
  grad_abs_max = jax.tree_util.tree_reduce(
      lambda x, y: jnp.maximum(x, jnp.max(jnp.abs(y))), grad, initializer=0) # max(max(x1,x2),x3)...

  grad_norm = tree_norm(grad)

  # 限制梯度的 L2 范数，如果配置了梯度范数上限，将梯度的 L2 范数限制在此范围内。
  if config.grad_max_norm > 0:
    mult = jnp.minimum(
        1, config.grad_max_norm / (jnp.finfo(jnp.float32).eps + grad_norm))
    grad = jax.tree_map(lambda z: mult * z, grad)
  grad_norm_clipped = tree_norm(grad)


  # 应用梯度更新，使用梯度和学习率更新优化器状态，然后生成新的模型状态。
  new_optimizer = state.optimizer.apply_gradient(
      grad, learning_rate=learning_rate)
  new_state = state.replace(optimizer=new_optimizer)

  # 计算 PSNR 和生成训练统计，这部分计算平均信噪比（PSNR），然后创建一个 TrainStats 数据对象来存储当前训练步骤的各种统计数据。
  psnrs = math.mse_to_psnr(losses)
  stats = TrainStats(
      loss=loss,
      losses=losses,
      losses_georeg=losses_georeg,
      disp_mses=disp_mses,
      normal_maes=normal_maes,
      weight_l2=weight_l2,
      psnr=psnrs[-1],
      psnrs=psnrs,
      grad_norm=grad_norm,
      grad_abs_max=grad_abs_max,
      grad_norm_clipped=grad_norm_clipped,
      all_depth_loss =all_depth_loss,
  )

  return new_state, stats, rng


# 作为程序的入口点，这个函数负责整个训练流程的设置和执行。包括随机种子的设置、数据集和模型的加载、优化器的配置、并行训练的设置、训练循环的执行、测试集评估和模型检查点的保存。
def main(unused_argv):

  # 这行代码使用固定的种子初始化一个 JAX 随机数生成器 (PRNGKey)，确保实验的可重复性。
  rng = random.PRNGKey(20200823)
  # Shift the numpy random seed by host_id() to shuffle data loaded by different
  # hosts.
  
  # 为 NumPy 随机操作设置种子，其中 jax.host_id() 确保在多主机设置中，每台机器得到不同的种子。
  np.random.seed(20201473 + jax.host_id())

  # 加载包含所有训练参数和设置的配置文件。
  config = configs.load_config()

  # 确保每个设备都能获得整数个数据样本，这对于并行训练是必要的。
  if config.batch_size % jax.device_count() != 0:
    raise ValueError('Batch size must be divisible by the number of devices.')

  # 加载训练和测试数据集。
  dataset = datasets_depth_llff_dtu.load_dataset('train', config.data_dir, config)
  test_dataset = datasets_depth_llff_dtu.load_dataset('test', config.data_dir, config)

  # 模型和变量的初始化,分割随机数生成器，以便为模型初始化提供独立的随机种子。然后构建模型和初始化参数。
  rng, key = random.split(rng)
  model, variables = models.construct_mipnerf(
      key,
      dataset.peek()['rays'],
      config,
  )

  # 计算模型参数数量
  num_params = jax.tree_util.tree_reduce(
      lambda x, y: x + jnp.prod(jnp.array(y.shape)), variables, initializer=0)
  print(f'Number of parameters being optimized: {num_params}')

  # 创建优化器,使用 Adam 优化器初始化，并创建包含优化器状态的 TrainState 对象。
  optimizer = flax.optim.Adam(config.lr_init).create(variables)
  state = utils.TrainState(optimizer=optimizer)
  del optimizer, variables

  # 准备并行训练的训练步,使用 jax.pmap（并行映射）来准备训练步，允许在多个 GPU 上同时运行
  train_pstep = jax.pmap(
      functools.partial(train_step, model, config), axis_name='batch',
      in_axes=(0, 0, 0, None, None, None, None))

  # Because this is only used for test set rendering, we disable randomization
  # and use the "final" padding for resampling.

  # 这个函数用于评估阶段的渲染，它在确定性模式下应用模型到给定的光线数据上，并收集所有设备上的结果。
  # 定义评估函数，并用 jax.pmap 并行化。这允许在多个设备上进行渲染评估
  def render_eval_fn(variables, _, rays):
    return jax.lax.all_gather(
        model.apply(
            variables,
            None,  # Deterministic.
            rays,
            resample_padding=config.resample_padding_final,
            compute_extras=True), axis_name='batch')

  render_eval_pfn = jax.pmap(
      render_eval_fn,
      axis_name='batch',
      in_axes=(None, None, 0),  # Only distribute the data input.
      donate_argnums=(3,),
  )


  # 计算两个图像之间的结构相似性指数 (SSIM)，这是图像质量评估中常用的指标。
  def ssim_fn(x, y):
    return structural_similarity(x, y, multichannel=True)

  # 设置目录和恢复检查点
  if not utils.isdir(config.checkpoint_dir):
    utils.makedirs(config.checkpoint_dir)
  state = checkpoints.restore_checkpoint(config.checkpoint_dir, state)


  # Resume training at the step of the last checkpoint.
  # 设置训练的开始步骤
  init_step = state.optimizer.state.step + 1
  state = flax.jax_utils.replicate(state)

  # 设置 TensorBoard,如果是主机，则初始化 TensorBoard 写入器，并记录配置。
  if jax.host_id() == 0:
    summary_writer = tensorboard.SummaryWriter(config.checkpoint_dir)
    summary_writer.text('config', f'<pre>{config}</pre>', step=0)

  # Prefetch_buffer_size = 3 x batch_size
  # 数据预取, 将数据预取到设备上，以减少数据加载的时间。
  pdataset = flax.jax_utils.prefetch_to_device(dataset, 3)

  # 禁用垃圾收集并启动训练,禁用 Python 的垃圾收集以提高效率，为每个设备分配一个随机数生成器，并开始训练循环
  rng = rng + jax.host_id()  # Make random seed separate across hosts.
  rngs = random.split(rng, jax.local_device_count())  # For pmapping RNG keys.
  gc.disable()  # Disable automatic garbage collection for efficiency.

  # 初始化训练监控变量,这些行初始化用于跟踪训练性能和统计的变量，例如总时间、总步数和平均 PSNR 的分子和分母。train_start_time 记录了当前时间以计算训练阶段的耗时。
  total_time = 0
  total_steps = 0
  avg_psnr_numer = 0.
  avg_psnr_denom = 0
  train_start_time = time.time()
  
  # 这里开始了训练循环，通过迭代从预处理的数据集(pdataset)中获取批次数据
  for step, batch in zip(range(init_step, config.max_steps + 1), pdataset):

    # 动态调整学习率和重采样填充,这些行计算每步的学习率和重采样填充。学习率根据预设的衰减策略动态调整，而重采样填充则在训练过程中线性插值。
    learning_rate = math.learning_rate_decay(
        step,
        config.lr_init,
        config.lr_final,
        config.max_steps,
        config.lr_delay_steps,
        config.lr_delay_mult,
    )

    resample_padding = math.log_lerp(
        step / config.max_steps,
        config.resample_padding_init,
        config.resample_padding_final,
    )

    # 计算 TV-Norm 损失权重,根据配置，计算用于 TV-Norm 正则化的损失权重，这有助于维持训练的稳定性和防止过拟合。
    if config.depth_tvnorm_decay: #True
      tvnorm_loss_weight = math.compute_tvnorm_weight( #1-i/max_step
          step, config.depth_tvnorm_maxstep, #512
          config.depth_tvnorm_loss_mult_start, #400
          config.depth_tvnorm_loss_mult_end) #0.1
    else:
      tvnorm_loss_weight = config.depth_tvnorm_loss_mult

    # 执行一步训练,在并行设备上执行一步训练，更新模型状态，获取训练统计，并更新随机数生成器状态。
    state, stats, rngs = train_pstep(
        rngs,
        state,
        batch,
        learning_rate,
        resample_padding,
        tvnorm_loss_weight,
        step,
    )

    ########################################################################################
    # 根据配置定期执行垃圾回收，以管理内存使用。
    if step % config.gc_every == 0:
      gc.collect()  # Disable automatic garbage collection for efficiency.

    # Log training summaries. This is put behind a host_id check because in
    # multi-host evaluation, all hosts need to run inference even though we
    # only use host 0 to record results.

    # 记录训练摘要和性能统计,这部分代码负责在训练过程中定期记录和打印关键性能指标和损失统计，仅在主机上执行。使用 TensorBoard 记录器(summary_writer)来记录各种训练指标。
    if jax.host_id() == 0:
      avg_psnr_numer += stats.psnr[0]
      avg_psnr_denom += 1
      if step % config.print_every == 0:
        elapsed_time = time.time() - train_start_time
        steps_per_sec = config.print_every / elapsed_time
        rays_per_sec = config.batch_size * steps_per_sec

        # A robust approximation of total training time, in case of pre-emption.
        total_time += int(round(TIME_PRECISION * elapsed_time))
        total_steps += config.print_every
        approx_total_time = int(round(step * total_time / total_steps))

        avg_psnr = avg_psnr_numer / avg_psnr_denom
        avg_psnr_numer = 0.
        avg_psnr_denom = 0
        

        # For some reason, the `stats` object has a superfluous dimension.
        stats = jax.tree_map(lambda x: x[0], stats)
        summary_writer.scalar('num_params', num_params, step)
        summary_writer.scalar('train_loss', stats.loss, step)
        summary_writer.scalar('train_psnr', stats.psnr, step)
        if config.compute_disp_metrics:
          for i, disp_mse in enumerate(stats.disp_mses):
            summary_writer.scalar(f'train_disp_mse_{i}', disp_mse, step)
        if config.compute_normal_metrics:
          for i, normal_mae in enumerate(stats.normal_maes):
            summary_writer.scalar(f'train_normal_mae_{i}', normal_mae, step)
        summary_writer.scalar('train_avg_psnr', avg_psnr, step)
        summary_writer.scalar('train_avg_psnr_timed', avg_psnr,
                              total_time // TIME_PRECISION)
        summary_writer.scalar('train_avg_psnr_timed_approx', avg_psnr,
                              approx_total_time // TIME_PRECISION)
        for i, l in enumerate(stats.losses):
          summary_writer.scalar(f'train_losses_{i}', l, step)
        for i, l in enumerate(stats.losses_georeg):
          summary_writer.scalar(f'train_losses_depth_tv_norm{i}', l, step)
        for i, p in enumerate(stats.psnrs):
          summary_writer.scalar(f'train_psnrs_{i}', p, step)
        summary_writer.scalar('weight_l2', stats.weight_l2, step)
        summary_writer.scalar('train_grad_norm', stats.grad_norm, step)
        summary_writer.scalar('train_grad_norm_clipped',
                              stats.grad_norm_clipped, step)
        summary_writer.scalar('train_grad_abs_max', stats.grad_abs_max, step)
        summary_writer.scalar('learning_rate', learning_rate, step)
        summary_writer.scalar('tvnorm_loss_weight', tvnorm_loss_weight, step)
        summary_writer.scalar('resample_padding', resample_padding, step)
        summary_writer.scalar('train_steps_per_sec', steps_per_sec, step)
        summary_writer.scalar('train_rays_per_sec', rays_per_sec, step)
        precision = int(np.ceil(np.log10(config.max_steps))) + 1
        print(f'{step:{precision}d}' + f'/{config.max_steps:d}: ' +
              f'loss={stats.loss:0.4f}, ' + f'avg_psnr={avg_psnr:0.2f}, ' +
              f'weight_l2={stats.weight_l2:0.2e}, ' +
              f'lr={learning_rate:0.2e}, '
              f'pad={resample_padding:0.2e}, ' +
              f'{rays_per_sec:0.0f} rays/sec')
        train_start_time = time.time()

      # 保存检查点,定期保存模型的状态，确保在训练中断时可以从检查点恢复。
      if step % config.checkpoint_every == 0:
        state_to_save = jax.device_get(jax.tree_map(lambda x: x[0], state))
        checkpoints.save_checkpoint(
            config.checkpoint_dir, state_to_save, int(step), keep=100)

    # Test-set evaluation.

    # 测试集评估,在指定的训练步骤进行测试集评估，生成和可视化渲染图像，用于监控模型在未见数据上的表现。
    if config.train_render_every > 0 and step % config.train_render_every == 0:
      # We reuse the same random number generator from the optimization step
      # here on purpose so that the visualization matches what happened in
      # training.
      eval_start_time = time.time()
      eval_variables = jax.device_get(jax.tree_map(lambda x: x[0],
                                                   state)).optimizer.target
      test_case = next(test_dataset)
      rendering = models.render_image(
          functools.partial(render_eval_pfn, eval_variables),
          test_case['rays'],
          rngs[0],
          config)

      vis_start_time = time.time()
      vis_suite = vis.visualize_suite(rendering, test_case['rays'], config)
      print(f'Visualized in {(time.time() - vis_start_time):0.3f}s')

      # Log eval summaries on host 0.

      # 测试评估摘要,在主机上记录测试评估的结果，包括 PSNR 和 SSIM 指标，以及渲染的图像
      if jax.host_id() == 0:
        if not config.render_path:
          psnr = float(
              math.mse_to_psnr(((
                  rendering['rgb'] - test_case['rgb'])**2).mean()))
          ssim = float(ssim_fn(rendering['rgb'], test_case['rgb']))
        eval_time = time.time() - eval_start_time
        num_rays = jnp.prod(jnp.array(test_case['rays'].directions.shape[:-1]))
        rays_per_sec = num_rays / eval_time
        summary_writer.scalar('test_rays_per_sec', rays_per_sec, step)
        print(f'Eval {step}: {eval_time:0.3f}s., {rays_per_sec:0.0f} rays/sec')
        if not config.render_path:
          print(f'PSNR={psnr:.4f} SSIM={ssim:.4f}')
          summary_writer.scalar('test_psnr', psnr, step)
          summary_writer.scalar('test_ssim', ssim, step)
          summary_writer.image('test_target', test_case['rgb'], step)

        for k, v in vis_suite.items():
          if k=='line_rays':
            for i in range(v.shape[0]):
              summary_writer.scalar('ray weights', v[i],i)
          elif k=='line_rgbs':
            for i in range(v.shape[0]):
              summary_writer.scalar('ray rgbs', v[i],i)
          else:
            summary_writer.image('test_pred_' + k, v, step)

  # 末尾的检查点保存,确保在训练结束时保存最终状态，即使最后一个步骤不恰好是检查点保存步骤。
  if config.max_steps % config.checkpoint_every != 0:
    state = jax.device_get(jax.tree_map(lambda x: x[0], state))
    checkpoints.save_checkpoint(
        config.checkpoint_dir, state, int(config.max_steps), keep=100)

# 启动点,这是 Python 脚本的典型入口点，当直接运行该脚本时，main 函数将被执行。这确保了脚本作为独立程序运行时的正确行为。
if __name__ == '__main__':
  app.run(main)

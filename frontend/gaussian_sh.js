// Real spherical harmonics follow the degree 0–3 basis in the pinned original
// gaussian-splatting/utils/sh_utils.py. Coefficients are channel-major, including
// DC. Input direction points from the camera toward the Gaussian in world space.
export const SH_C0 = 0.28209479177387814;
const C1 = 0.4886025119029199;
const C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396];
const C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435];

export function validateSH(sh, degree) {
  if ((!Array.isArray(sh) && !ArrayBuffer.isView(sh)) || !Number.isInteger(degree) || degree < 0 || degree > 3) {
    throw new Error('SH 必须提供 0–3 阶 sh_degree 和按 RGB 通道排列的系数。');
  }
  const stride = sh.length / 3;
  if (![1, 4, 9, 16].includes(stride) || stride < (degree + 1) ** 2) {
    throw new Error('SH 系数数量或数值无效；需要每个 RGB 通道的完整有限系数。');
  }
  for (const value of sh) if (!Number.isFinite(value) || !Number.isFinite(Math.fround(value))) {
    throw new Error('SH 系数数量或数值无效；需要每个 RGB 通道的完整有限系数。');
  }
  return stride;
}

export function shBasis(direction, degree = 3) {
  if (!Number.isInteger(degree) || degree < 0 || degree > 3 || direction.length !== 3 || !Array.from(direction).every(Number.isFinite)) {
    throw new Error('SH 方向必须是有限三维向量，阶数必须为 0–3。');
  }
  const length = Math.hypot(...direction);
  if (!(length > 0)) throw new Error('SH 方向不能为零向量。');
  const [x, y, z] = Array.from(direction, value => value / length);
  const basis = [SH_C0];
  if (degree >= 1) basis.push(-C1 * y, C1 * z, -C1 * x);
  if (degree >= 2) {
    const xx = x*x, yy = y*y, zz = z*z;
    basis.push(C2[0]*x*y, C2[1]*y*z, C2[2]*(2*zz-xx-yy), C2[3]*x*z, C2[4]*(xx-yy));
    if (degree >= 3) basis.push(C3[0]*y*(3*xx-yy), C3[1]*x*y*z, C3[2]*y*(4*zz-xx-yy),
      C3[3]*z*(2*zz-3*xx-3*yy), C3[4]*x*(4*zz-xx-yy), C3[5]*z*(xx-yy), C3[6]*x*(xx-3*yy));
  }
  return basis;
}

export function evaluateSH(sh, degree, direction) {
  const stride = validateSH(sh, degree), basis = shBasis(direction, degree);
  return [0, 1, 2].map(channel => Math.max(0, .5 + basis.reduce((sum, value, index) => sum + value * sh[channel*stride+index], 0)));
}

export function gaussianLoadPlan(count, maxGaussians = null) {
  if (!Number.isSafeInteger(count) || count < 0) throw new Error('Invalid Gaussian count');
  if (maxGaussians != null && (!Number.isSafeInteger(maxGaussians) || maxGaussians < 1)) {
    throw new Error('显式查看预算 maxGaussians 必须为正整数。');
  }
  const selectedCount = maxGaussians == null ? count : Math.min(count, maxGaussians);
  return {total: count, selectedCount, maxGaussians, sampled: selectedCount < count,
    omittedByBudget: count-selectedCount,
    *indices() { for (let index = 0; index < selectedCount; index++) yield Math.floor(index*count/selectedCount); }};
}

export function shTextureLayout(count, coefficientStride = 16, maxTextureSize = 4096) {
  if (!Number.isInteger(maxTextureSize) || maxTextureSize < 16) throw new Error('GPU SH 纹理尺寸不足。');
  if (!Number.isSafeInteger(count) || count < 0 || count > 16777216) throw new Error('SH 记录超过浮点索引可精确表示的范围。');
  if (![1,4,9,16].includes(coefficientStride)) throw new Error('SH 纹理系数数量无效。');
  const pixels = Math.max(1, count * coefficientStride);
  const width = count ? Math.min(maxTextureSize, Math.max(16, Math.ceil(Math.sqrt(pixels)/16)*16)) : 1;
  const height = Math.ceil(pixels / width);
  if (height > maxTextureSize) throw new Error(`当前 GPU 无法容纳 ${count.toLocaleString()} 个高斯的完整 SH 系数（纹理尺寸上限 ${maxTextureSize}）。没有抽样或降低球谐阶数，请换用容量更大的图形设备。`);
  return {width,height,count,coefficientStride,bytes:width*height*16};
}

export function packSHTexture(records, maxTextureSize = 4096) {
  let coefficientStride=1;
  for(const record of records)coefficientStride=Math.max(coefficientStride,validateSH(record.sh,record.sh_degree));
  const layout=shTextureLayout(records.length,coefficientStride,maxTextureSize);
  let data;
  try{data=new Float32Array(layout.bytes/4);}catch{throw new Error('内存不足，无法载入完整 SH 系数；没有自动抽样或降低球谐阶数。');}
  records.forEach((record, index) => {
    const stride = record.sh.length/3;
    // Keep every supplied coefficient, including inactive higher bands. The
    // shader still uses each record's real active degree; SH0 never becomes SH3.
    for (let basis = 0; basis < stride; basis++) {
      const offset = (index*coefficientStride+basis)*4;
      data[offset] = record.sh[basis];
      data[offset+1] = record.sh[stride+basis];
      data[offset+2] = record.sh[2*stride+basis];
    }
  });
  return {...layout,data};
}

// Texture positions stay in source record order. Sorting updates only the small
// per-instance shInfo=(record index, active degree), never the coefficient bank.
export const gaussianSHShader = `
attribute vec2 shInfo;
uniform highp sampler2D shTexture;
uniform int shTextureWidth;
uniform int shCoefficientStride;
vec3 shCoefficient(int index, int basis) {
  int address = index * shCoefficientStride + basis;
  return texelFetch(shTexture, ivec2(address % shTextureWidth, address / shTextureWidth), 0).rgb;
}
vec3 evaluateGaussianSH(vec3 direction, int index, int degree) {
  float x=direction.x, y=direction.y, z=direction.z;
  vec3 result = ${SH_C0} * shCoefficient(index, 0);
  if (degree > 0) {
    result += -${C1}*y*shCoefficient(index,1) + ${C1}*z*shCoefficient(index,2) - ${C1}*x*shCoefficient(index,3);
    if (degree > 1) {
      float xx=x*x, yy=y*y, zz=z*z;
      result += ${C2[0]}*x*y*shCoefficient(index,4) + (${C2[1]})*y*z*shCoefficient(index,5)
              + ${C2[2]}*(2.0*zz-xx-yy)*shCoefficient(index,6) + (${C2[3]})*x*z*shCoefficient(index,7)
              + ${C2[4]}*(xx-yy)*shCoefficient(index,8);
      if (degree > 2) {
        result += (${C3[0]})*y*(3.0*xx-yy)*shCoefficient(index,9) + ${C3[1]}*x*y*z*shCoefficient(index,10)
                + (${C3[2]})*y*(4.0*zz-xx-yy)*shCoefficient(index,11) + ${C3[3]}*z*(2.0*zz-3.0*xx-3.0*yy)*shCoefficient(index,12)
                + (${C3[4]})*x*(4.0*zz-xx-yy)*shCoefficient(index,13) + ${C3[5]}*z*(xx-yy)*shCoefficient(index,14)
                + (${C3[6]})*x*(xx-3.0*yy)*shCoefficient(index,15);
      }
    }
  }
  return max(vec3(0.0), result + vec3(0.5));
}`;

// Exact, stable IEEE-754 float64 ordering. Unlike bucketed depth sort, this
// retains every depth bit and never merges nearby Gaussians into one bin.
const LITTLE_ENDIAN=new Uint8Array(new Uint32Array([1]).buffer)[0]===1;
const LOW=LITTLE_ENDIAN?0:1,HIGH=LITTLE_ENDIAN?1:0;

export function sortDepthIndices(order,depth,scratch){
  if(!(order instanceof Uint32Array)||!(depth instanceof Float64Array)||!(scratch instanceof Uint32Array)||scratch.length<order.length)
    throw new Error('高斯深度排序缓冲区无效。');
  const words=new Uint32Array(depth.buffer,depth.byteOffset,depth.length*2),counts=new Uint32Array(256);
  for(const index of order)if(index>=depth.length||Number.isNaN(depth[index]))throw new Error('高斯深度排序索引或深度无效。');
  let input=order,output=scratch.subarray(0,order.length);
  for(let pass=0;pass<8;pass++){
    const shift=(pass%4)*8,highWord=pass>=4;
    const digit=index=>{
      const low=words[index*2+LOW];let high=words[index*2+HIGH];
      if(low===0&&(high&0x7fffffff)===0)high=0; // Keep +0/-0 ties stable.
      const value=highWord?high:low;
      const ordered=(high&0x80000000)?~value:(highWord?value^0x80000000:value);
      return (ordered>>>shift)&255;
    };
    counts.fill(0);for(const index of input)counts[digit(index)]++;
    let offset=0;for(let i=0;i<256;i++){const count=counts[i];counts[i]=offset;offset+=count;}
    for(const index of input)output[counts[digit(index)]++]=index;
    [input,output]=[output,input];
  }
  return order;
}

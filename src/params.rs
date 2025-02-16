use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}
/* 
impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        todo!("实现从safetensors文件的模型参数加载");
        // let get_tensor: impl Fn(&str) -> Tensor<f32> = |name: &str| {
        // ...    
        // };
        
        // LLamaParams {
        //     embedding_table: get_tensor(...),
        //     ...
        // }
    }
}
*/
impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Result<Self, String> {
        let get_tensor = |name: &str| -> Result<Tensor<f32>, String> {
            match safetensor.tensor(name) {
                Ok(tensor) => {
                    let data = tensor.data().to_vec();
                    let data_f32: Vec<f32> = data.chunks_exact(4).map(|chunk| {
                        f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                    }).collect();
                    let shape = tensor.shape().to_vec();
                    Ok(Tensor::new(data_f32, &shape))
                }
                Err(_) => Err(format!("Tensor {} not found", name)),
            }
        };

        let n_layers = config.num_hidden_layers;
        let mut rms_att_w = Vec::with_capacity(n_layers);
        let mut wq = Vec::with_capacity(n_layers);
        let mut wk = Vec::with_capacity(n_layers);
        let mut wv = Vec::with_capacity(n_layers);
        let mut wo = Vec::with_capacity(n_layers);
        let mut rms_ffn_w = Vec::with_capacity(n_layers);
        let mut w_up = Vec::with_capacity(n_layers);
        let mut w_gate = Vec::with_capacity(n_layers);
        let mut w_down = Vec::with_capacity(n_layers);

        for layer in 0..n_layers {
            rms_att_w.push(get_tensor(&format!("model.layers.{}.input_layernorm.weight", layer))?);
            wq.push(get_tensor(&format!("model.layers.{}.self_attn.q_proj.weight", layer))?);
            wk.push(get_tensor(&format!("model.layers.{}.self_attn.k_proj.weight", layer))?);
            wv.push(get_tensor(&format!("model.layers.{}.self_attn.v_proj.weight", layer))?);
            wo.push(get_tensor(&format!("model.layers.{}.self_attn.o_proj.weight", layer))?);
            rms_ffn_w.push(get_tensor(&format!("model.layers.{}.post_attention_layernorm.weight", layer))?);
            w_up.push(get_tensor(&format!("model.layers.{}.mlp.up_proj.weight", layer))?);
            w_gate.push(get_tensor(&format!("model.layers.{}.mlp.gate_proj.weight", layer))?);
            w_down.push(get_tensor(&format!("model.layers.{}.mlp.down_proj.weight", layer))?);
        }

        let embedding_table = get_tensor("model.embed_tokens.weight")?;
        
        // 手动复制embedding_table
        let lm_head = if config.tie_word_embeddings {
            // 直接创建一个新的Tensor对象作为lm_head
            Tensor::new(embedding_table.data().to_vec(), &embedding_table.shape())
        } else {
            get_tensor("lm_head.weight")?
        };

        let rms_out_w = get_tensor("model.norm.weight")?;

        Ok(LLamaParams {
            embedding_table,
            rms_att_w,
            wq,
            wk,
            wv,
            wo,
            rms_ffn_w,
            w_up,
            w_gate,
            w_down,
            rms_out_w,
            lm_head,
        })
    }
}

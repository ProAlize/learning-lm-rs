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
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // 定义一个从safetensor中读取Tensor的辅助函数
        let get_tensor = |name: &str| -> Tensor<f32> {
            let tensor_data = safetensor.tensor(name).unwrap(); // 从safetensor中获取tensor
            let shape = tensor_data.shape().clone(); // 获取tensor的形状
            Tensor::new(tensor_data.to_vec(), &shape) // 使用数据和形状创建Tensor
        };

        // 从safetensor中加载各个参数
        let embedding_table = get_tensor("embedding_table");
        let rms_att_w = (0..config.num_hidden_layers)
            .map(|i| get_tensor(&format!("rms_att_w_{}", i)))
            .collect::<Vec<Tensor<f32>>>();
        let wq = (0..config.num_hidden_layers)
            .map(|i| get_tensor(&format!("wq_{}", i)))
            .collect::<Vec<Tensor<f32>>>();
        let wk = (0..config.num_hidden_layers)
            .map(|i| get_tensor(&format!("wk_{}", i)))
            .collect::<Vec<Tensor<f32>>>();
        let wv = (0..config.num_hidden_layers)
            .map(|i| get_tensor(&format!("wv_{}", i)))
            .collect::<Vec<Tensor<f32>>>();
        let wo = (0..config.num_hidden_layers)
            .map(|i| get_tensor(&format!("wo_{}", i)))
            .collect::<Vec<Tensor<f32>>>();
        let rms_ffn_w = (0..config.num_hidden_layers)
            .map(|i| get_tensor(&format!("rms_ffn_w_{}", i)))
            .collect::<Vec<Tensor<f32>>>();
        let w_up = (0..config.num_hidden_layers)
            .map(|i| get_tensor(&format!("w_up_{}", i)))
            .collect::<Vec<Tensor<f32>>>();
        let w_gate = (0..config.num_hidden_layers)
            .map(|i| get_tensor(&format!("w_gate_{}", i)))
            .collect::<Vec<Tensor<f32>>>();
        let w_down = (0..config.num_hidden_layers)
            .map(|i| get_tensor(&format!("w_down_{}", i)))
            .collect::<Vec<Tensor<f32>>>();
        let rms_out_w = get_tensor("rms_out_w");
        let lm_head = get_tensor("lm_head");

        // 如果配置中的tie_word_embeddings为true，则共享embedding矩阵
        let embedding_table = if config.tie_word_embeddings {
            let lm_head_data = lm_head.clone(); // 使用lm_head的权重作为embedding
            embedding_table.set_data(lm_head_data.data().to_vec());
            embedding_table
        } else {
            embedding_table
        };

        LLamaParams {
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
        }
    }
}

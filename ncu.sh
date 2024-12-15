ncu --metrics \
launch__registers_per_thread,\
launch__occupancy_per_register_count,\
launch__occupancy_per_block_size,\
launch__occupancy_limit_warps,\
launch__occupancy_limit_registers,\
launch__occupancy_limit_blocks,\
launch__block_size,\
launch__grid_size,\
sm__warps_active.avg,\
sm__maximum_warps_avg_per_active_cycle,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
sm__sass_thread_inst_executed_per_thread.avg,\
sm__sass_average_register_used_per_thread.avg,\
sm__sass_maximum_register_used_per_thread.avg,\
sm__warps_active.avg,\
sm__ctas_active.avg,\
sm__ctas_launched.avg,\
sm__occupancy.avg,\
launch__waves_per_multiprocessor.avg,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_cycles_active.avg,\
sm__cycles_elapsed.avg,\
gpu__time_duration.avg,\
sm__inst_executed_pipe_alu.avg,\
sm__inst_executed_pipe_fma.avg,\
sm__inst_executed_pipe_fp16.avg,\
sm__pipe_alu_cycles_active.avg,\
sm__pipe_fma_cycles_active.avg,\
sm__pipe_fp16_cycles_active.avg,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum,\
lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_st.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum \
-k gemv_fp16 \
python czn_test.py
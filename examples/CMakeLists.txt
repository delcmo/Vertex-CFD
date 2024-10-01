set(REGRESSION_XML_FILES
    inputs/incompressible/incompressible_2d_backward_facing_step.xml
    inputs/incompressible/incompressible_2d_channel.xml
    inputs/incompressible/incompressible_2d_channel_periodic.xml
    inputs/incompressible/incompressible_3d_channel_periodic.xml
    inputs/incompressible/incompressible_oscillating_laminar_flow_2d.xml
    inputs/incompressible/incompressible_blunt_plate_laminar_flow_2d.xml
    inputs/incompressible/incompressible_2d_heated_channel.xml
    inputs/incompressible/incompressible_2d_rotating_cylinder_viscous.xml
    inputs/incompressible/incompressible_2d_concentric_cylinder_convection.xml
    inputs/incompressible/incompressible_2d_taylor_green_vortex.xml
    inputs/incompressible/incompressible_2d_tee_junction.xml
    inputs/incompressible/incompressible_2d_planar_poiseuille.xml
    inputs/incompressible/incompressible_2d_planar_poiseuille_cuda.xml
    inputs/incompressible/incompressible_2d_standard_k_epsilon_turbulence_channel.xml
    inputs/incompressible/incompressible_2d_realizable_k_epsilon_turbulence_channel.xml
    inputs/incompressible/incompressible_2d_spalart_allmaras_turbulence_channel.xml
    inputs/incompressible/incompressible_2d_spalart_allmaras_turbulence_heated_channel.xml
    inputs/incompressible/incompressible_3d_wale_cavity.xml
    inputs/induction_less_mhd/mhd_2d_hartmann_pb_periodic_insulating.xml
    inputs/full_induction_mhd/full_induction_vortex_2d_pb.xml
    inputs/full_induction_mhd/divergence_advection_2d.xml
    inputs/full_induction_mhd/current_sheet_2d.xml
    inputs/full_induction_mhd/ldc_2d_bx_010.xml
    inputs/full_induction_mhd/ldc_2d_mixed_b_050_rotated.xml
    )
file(COPY ${REGRESSION_XML_FILES} DESTINATION .)

set(REGRESSION_EXO_FILES
    mesh/incompressible/2d_backward_facing_step.exo
    mesh/incompressible/pipe_hex.exo
    mesh/incompressible/2d_cyclinder_vertex_tri.exo
    mesh/incompressible/bluntplate_square.exo
    mesh/incompressible/2d_concentric_cylinders_rad10.exo
    mesh/incompressible/2d_concentric_cylinders_rad20.exo
    mesh/incompressible/2d_concentric_cylinders_rad40.exo
    mesh/incompressible/2d_concentric_cylinders_rad80.exo
    mesh/incompressible/2d_concentric_convection.exo
    mesh/incompressible/turbulent_channel_mesh_one.exo
    mesh/incompressible/2d_tee_junction.exo
    mesh/full_induction_mhd/ldc_bl_41x41_30deg.exo
    mesh/full_induction_mhd/ldc_bl_81x81.exo
    )
file(COPY ${REGRESSION_EXO_FILES} DESTINATION .)

install(FILES
  inputs/incompressible/incompressible_2d_backward_facing_step.xml
  inputs/incompressible/incompressible_2d_channel.xml
  inputs/incompressible/incompressible_2d_channel_periodic.xml
  inputs/incompressible/incompressible_blunt_plate_laminar_flow_2d.xml
  inputs/incompressible/incompressible_2d_channel.xml
  inputs/incompressible/incompressible_oscillating_laminar_flow_2d.xml
  inputs/incompressible/incompressible_3d_channel_periodic.xml
  inputs/incompressible/incompressible_2d_rotating_cylinder_viscous.xml
  inputs/incompressible/incompressible_2d_concentric_cylinder_convection.xml
  inputs/incompressible/incompressible_2d_planar_poiseuille.xml
  inputs/incompressible/incompressible_2d_realizable_k_epsilon_turbulence_channel.xml
  inputs/incompressible/incompressible_2d_tee_junction.xml
  inputs/incompressible/incompressible_3d_wale_cavity.xml
  DESTINATION examples/incompressible)

install(FILES
  mesh/incompressible/2d_backward_facing_step.exo
  mesh/incompressible/2d_cyclinder_vertex_tri.exo
  mesh/incompressible/bluntplate_square.exo
  mesh/incompressible/pipe_hex.exo
  mesh/incompressible/2d_concentric_cylinders_rad40.exo
  mesh/incompressible/2d_concentric_convection.exo
  mesh/incompressible/2d_tee_junction.exo
  DESTINATION examples/incompressible)
